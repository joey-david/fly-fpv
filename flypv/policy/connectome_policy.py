"""Sparse connectome-constrained policy network.

The MaleCNS graph is used as a fixed sparse architecture, not as a claim about
real-time fly neural dynamics. Every control decision starts from zero activity,
encodes the current observation onto sensory populations, runs a small number of
message-passing iterations through the measured graph, and decodes motor-neuron
activity into flight controls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..connectome.circuits import FlightCircuit, MOTOR_GROUPS


@dataclass
class PolicyConfig:
    n_iters: int = 4
    learn_synapses: bool = False
    neuron_descriptors: int = 0
    act: str = "softplus"
    value_hidden: int = 256
    log_std_init: float = -1.6
    sensory_hidden: int = 64


def squashed_log_prob(dist, raw: torch.Tensor) -> torch.Tensor:
    logp = dist.log_prob(raw).sum(-1)
    return logp - (2 * (math.log(2) - raw - F.softplus(-2 * raw))).sum(-1)


def _act(name: str):
    return {"softplus": lambda x: F.softplus(x, beta=3.0),
            "tanh": torch.tanh,
            "relu": F.relu,
            "elu": F.elu}[name]


class ConnectomePolicy(nn.Module):
    def __init__(self, circuit: FlightCircuit, proprio_dim: int, action_dim: int,
                 cfg: PolicyConfig | None = None, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        self.circuit = circuit
        self.N = circuit.n
        self.action_dim = action_dim
        self.phi = _act(self.cfg.act)

        W = circuit.cx.W.tocsr().astype(np.float32)
        # Normalize aggregate input for stability while preserving the measured
        # relative synapse strengths into each neuron.
        row_abs = np.asarray(np.abs(W).sum(axis=1)).ravel()
        scale = 1.0 / np.maximum(row_abs, 1.0)
        Wn = (sp.diags(scale.astype(np.float32)) @ W).tocoo()

        order = np.lexsort((Wn.col, Wn.row))
        idx = torch.from_numpy(np.stack([Wn.row[order], Wn.col[order]]).astype(np.int64))
        val = torch.from_numpy(Wn.data[order].astype(np.float32))
        self.register_buffer("w_idx", idx)
        self.register_buffer("w_val", val)
        self.n_edges = val.numel()

        self.edge_gain = (nn.Parameter(torch.zeros(self.n_edges))
                          if self.cfg.learn_synapses else None)

        unknown = torch.from_numpy((circuit.cx.nt_sign == 0).astype(np.float32))
        self.register_buffer("sign_unknown", unknown)
        self.sign_logit = nn.Parameter(torch.zeros(self.N))
        if bool(unknown.any()):
            u = unknown.numpy()[Wn.col[order]] > 0
            with torch.no_grad():
                self.w_val[torch.from_numpy(u)] = self.w_val[torch.from_numpy(u)].abs()

        # Trainable node-wise transforms. `mix_logit` controls how aggressively
        # each message-passing iteration updates a node within this one decision.
        self.gain = nn.Parameter(torch.ones(self.N))
        self.bias = nn.Parameter(torch.zeros(self.N))
        self.mix_logit = nn.Parameter(torch.zeros(self.N))

        if self.cfg.neuron_descriptors:
            D = self.cfg.neuron_descriptors
            self.eta = nn.Parameter(torch.randn(self.N, D) * 0.05)
            self.neuron_mlp = nn.Sequential(
                nn.Linear(1 + D, 32), nn.SiLU(), nn.Linear(32, 1))
        else:
            self.eta = None

        # Vision is placed on its retinotopic connectome populations.
        self.retina_rows = torch.from_numpy(circuit.afferent.get("retina", np.array([], np.int64)))
        self.register_buffer("_ret", self.retina_rows)
        from ..env.vision import ON_TYPES
        col_of, on_of = [], []
        cols = sorted({(h1, h2, sd) for (h1, h2, sd) in circuit.hex_coords.values()})
        col_ix = {c: i for i, c in enumerate(cols)}
        types = circuit.cx.meta["type"].to_numpy()
        for r in self.retina_rows.tolist():
            key = circuit.hex_coords.get(int(r))
            col_of.append(col_ix.get(key, 0) if key else 0)
            on_of.append(1.0 if types[int(r)] in ON_TYPES else 0.0)
        self.register_buffer("ret_col", torch.tensor(col_of, dtype=torch.long))
        self.register_buffer("ret_is_on", torch.tensor(on_of, dtype=torch.float32))
        self.n_columns = len(cols)
        self.retina_gain = nn.Parameter(torch.tensor(1.0))

        # For flight performance rather than biological interpretation, the full
        # low-dimensional task/proprio vector is encoded into the nonvisual
        # afferents. This includes course geometry already present in the env obs.
        other = [v for k, v in circuit.afferent.items() if k != "retina" and len(v)]
        other_rows = np.concatenate(other) if other else np.array([], np.int64)
        self.register_buffer("_oth", torch.from_numpy(other_rows))
        h = self.cfg.sensory_hidden
        self.sensory_encoder = nn.Sequential(
            nn.Linear(proprio_dim, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.sensory_to_neurons = nn.Linear(h, max(len(other_rows), 1))
        nn.init.zeros_(self.sensory_to_neurons.bias)
        nn.init.normal_(self.sensory_to_neurons.weight, std=0.05)

        self.motor = MotorDecoder(circuit, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), self.cfg.log_std_init))
        self.value = nn.Sequential(
            nn.Linear(proprio_dim + 2 * len(MOTOR_GROUPS), self.cfg.value_hidden), nn.SiLU(),
            nn.Linear(self.cfg.value_hidden, self.cfg.value_hidden), nn.SiLU(),
            nn.Linear(self.cfg.value_hidden, 1))

        self.to(device)
        self.device = device

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(self, batch: int) -> torch.Tensor:
        """Compatibility helper for monitor/warm-start callers; state is not carried."""
        return torch.zeros(self.N, batch, device=self.device)

    def _W(self) -> torch.Tensor:
        val = self.w_val
        if self.edge_gain is not None:
            val = val * torch.exp(self.edge_gain.clamp(-2.0, 2.0))
        return torch.sparse_coo_tensor(
            self.w_idx, val, (self.N, self.N), is_coalesced=True, check_invariants=False
        )

    def presyn_sign(self) -> torch.Tensor:
        return (1.0 - self.sign_unknown
                + self.sign_unknown * torch.tanh(self.sign_logit)).unsqueeze(1)

    def forward(self, obs: dict, state: torch.Tensor | None = None):
        proprio = obs["proprio"]
        B = proprio.shape[0]
        drive = torch.zeros(self.N, B, device=proprio.device)

        if "on" in obs and len(self._ret):
            on = obs["on"][:, self.ret_col].T
            off = obs["off"][:, self.ret_col].T
            rd = self.ret_is_on.unsqueeze(1) * on + (1 - self.ret_is_on).unsqueeze(1) * off
            drive.index_add_(0, self._ret, rd * self.retina_gain)

        if len(self._oth):
            enc = self.sensory_encoder(proprio)
            drive.index_add_(0, self._oth, self.sensory_to_neurons(enc).T)

        # No temporal recurrence: each decision is an independent sparse network
        # evaluation. Iterations are depth/message passing within this decision.
        h = torch.zeros(self.N, B, device=proprio.device, dtype=proprio.dtype)
        W = self._W()
        sign = self.presyn_sign()
        mix = torch.sigmoid(self.mix_logit).unsqueeze(1)
        for _ in range(self.cfg.n_iters):
            m = torch.sparse.mm(W, h * sign) + drive
            z = self.gain.unsqueeze(1) * m + self.bias.unsqueeze(1)
            if self.eta is not None:
                z = z + self.neuron_mlp(
                    torch.cat([m.unsqueeze(-1),
                               self.eta.unsqueeze(1).expand(-1, B, -1)], -1)).squeeze(-1)
            h = (1 - mix) * h + mix * self.phi(z)

        mean, group_feat = self.motor(h)
        value = self.value(torch.cat([proprio, group_feat], -1)).squeeze(-1)
        return mean, self.log_std.expand_as(mean), value, h

    def dist(self, mean, log_std):
        return torch.distributions.Normal(mean, log_std.exp())

    def act(self, obs, state=None, deterministic: bool = False):
        mean, log_std, value, h = self.forward(obs)
        if deterministic:
            return torch.tanh(mean), None, value, h
        d = self.dist(mean, log_std)
        raw = d.rsample()
        return torch.tanh(raw), squashed_log_prob(d, raw), value, h, raw


class MotorDecoder(nn.Module):
    """Efferent activity -> 15 flight controls via anatomical motor groups."""

    WIRING: list[tuple[str, str, str, float]] = [
        ("power", "sum", "freq", 1.0),
        ("power", "sum", "amp_R", 0.7), ("power", "sum", "amp_L", 0.7),
        ("power", "diff", "amp_R", 0.6), ("power", "diff", "amp_L", -0.6),
        ("basalar", "sum", "amp_R", 0.8), ("basalar", "sum", "amp_L", 0.8),
        ("basalar", "diff", "amp_R", 1.0), ("basalar", "diff", "amp_L", -1.0),
        ("pterale", "sum", "offset_R", 0.9), ("pterale", "sum", "offset_L", 0.9),
        ("pterale", "diff", "offset_R", 0.8), ("pterale", "diff", "offset_L", -0.8),
        ("axillary", "sum", "stroke_tilt", 1.0),
        ("axillary", "diff", "dev_R", 0.7), ("axillary", "diff", "dev_L", -0.7),
        ("hinge", "sum", "aoa_down_R", 1.0), ("hinge", "sum", "aoa_down_L", 1.0),
        ("hinge", "diff", "aoa_down_R", 0.9), ("hinge", "diff", "aoa_down_L", -0.9),
        ("wing_misc", "sum", "aoa_up_R", 0.8), ("wing_misc", "sum", "aoa_up_L", 0.8),
        ("wing_misc", "diff", "aoa_up_R", 0.8), ("wing_misc", "diff", "aoa_up_L", -0.8),
        ("neck", "sum", "head_pitch", 1.0),
        ("neck", "diff", "head_yaw", 1.0),
        ("abdomen", "sum", "abdomen_pitch", 1.0),
    ]

    def __init__(self, circuit: FlightCircuit, action_dim: int):
        super().__init__()
        from ..wing import WING_CONTROLS
        self.groups = [g for g in MOTOR_GROUPS if len(circuit.efferent.get(g, []))]
        self.action_dim = action_dim

        rows, sides = [], []
        for g in self.groups:
            r = circuit.efferent[g]
            rows.append(torch.from_numpy(r))
            sides.append(torch.from_numpy(circuit.side_of(r).astype(np.float32)))
        for i, (r, s) in enumerate(zip(rows, sides)):
            self.register_buffer(f"rows_{i}", r)
            self.register_buffer(f"side_{i}", s)

        G = len(self.groups)
        self.readout = nn.Linear(2 * G, action_dim)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        gi = {g: i for i, g in enumerate(self.groups)}
        ci = {c: i for i, c in enumerate(WING_CONTROLS)}
        with torch.no_grad():
            for group, mode, ctrl, w in self.WIRING:
                if group in gi and ctrl in ci:
                    self.readout.weight[ci[ctrl], 2 * gi[group] + (0 if mode == "sum" else 1)] = w

        all_rows = torch.cat(rows) if rows else torch.zeros(0, dtype=torch.long)
        self.register_buffer("_all", all_rows)
        self.residual = nn.Linear(max(len(all_rows), 1), action_dim)
        nn.init.normal_(self.residual.weight, std=0.01)
        nn.init.zeros_(self.residual.bias)

    def forward(self, h: torch.Tensor):
        feats = []
        for i in range(len(self.groups)):
            rows = getattr(self, f"rows_{i}")
            side = getattr(self, f"side_{i}").unsqueeze(1)
            a = h[rows]
            feats.append(a.mean(0))
            denom = side.abs().sum().clamp(min=1.0)
            feats.append((a * side).sum(0) / denom)
        gf = torch.stack(feats, -1)
        return self.readout(gf) + self.residual(h[self._all].T), gf
