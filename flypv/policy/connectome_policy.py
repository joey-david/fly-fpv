"""The measured MaleCNS wiring diagram used as a recurrent policy network.

Connectivity remains fixed. Training tunes neuron intrinsic properties and the
small sensor/motor interfaces, while activity propagates through measured sparse
edges. Synapse counts are softly normalised: a typical neuron receives unit
aggregate drive, but unusually strong anatomical convergence is no longer erased
entirely as it was by strict row normalisation.
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

# Indices into HoopRaceEnv.PROPRIO_DIM. Navigation/gate geometry deliberately
# appears nowhere here: the actor can discover the course through vision, rather
# than leaking privileged gate coordinates through every mechanosensory neuron.
SENSOR_FEATURES: dict[str, tuple[int, ...]] = {
    "haltere": (0, 1, 2, 9, 10),                    # body rate + wing phase
    "johnston": (6, 7, 8),                          # body-frame airspeed
    "wing_cs": tuple(range(9, 23)),                  # phase + prior wing controls
    "hairplate": (3, 4, 5, 23, 24),                 # gravity/head orientation proxy
    "chordotonal": tuple(range(9, 26)),              # phase + previous body controls
}


@dataclass
class PolicyConfig:
    n_iters: int = 4
    carry: float = 0.5
    learn_synapses: bool = False
    neuron_descriptors: int = 0
    act: str = "softplus"
    value_hidden: int = 256
    log_std_init: float = -1.6
    sensory_hidden: int = 32
    input_norm_power: float = 0.5  # 1=strict row norm, 0=retain relative convergence


def squashed_log_prob(dist, raw: torch.Tensor) -> torch.Tensor:
    logp = dist.log_prob(raw).sum(-1)
    return logp - (2 * (math.log(2) - raw - F.softplus(-2 * raw))).sum(-1)


def _act(name: str):
    return {"softplus": lambda x: F.softplus(x, beta=3.0),
            "tanh": torch.tanh,
            "relu": F.relu,
            "elu": F.elu}[name]


def _soft_row_normalize(W: sp.csr_matrix, power: float) -> sp.csr_matrix:
    """Stabilise sparse drive without deleting anatomical convergence strength.

    ``power=1`` reproduces strict row normalisation. At the default 0.5, total
    input magnitude scales with sqrt(total synapse count / median), clipped to a
    conservative [0.25, 4] envelope. Relative synapse weights within each row are
    always preserved.
    """
    if not 0.0 <= power <= 1.0:
        raise ValueError("input_norm_power must be in [0, 1]")
    row_abs = np.asarray(np.abs(W).sum(axis=1)).ravel().astype(np.float64)
    nz = row_abs > 0
    if not nz.any():
        return W.copy()
    ref = float(np.median(row_abs[nz]))
    target = np.ones_like(row_abs)
    target[nz] = np.clip((row_abs[nz] / max(ref, 1.0)) ** (1.0 - power), 0.25, 4.0)
    scale = np.zeros_like(row_abs)
    scale[nz] = target[nz] / row_abs[nz]
    return (sp.diags(scale.astype(np.float32)) @ W).tocsr()


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
        Wn = _soft_row_normalize(W, self.cfg.input_norm_power).tocoo()
        order = np.lexsort((Wn.col, Wn.row))
        idx = torch.from_numpy(np.stack([Wn.row[order], Wn.col[order]]).astype(np.int64))
        val = torch.from_numpy(Wn.data[order].astype(np.float32))
        self.register_buffer("w_idx", idx)
        self.register_buffer("w_val", val)
        self.n_edges = val.numel()

        if self.cfg.learn_synapses:
            self.edge_gain = nn.Parameter(torch.zeros(self.n_edges))
        else:
            self.edge_gain = None

        unknown = torch.from_numpy((circuit.cx.nt_sign == 0).astype(np.float32))
        self.register_buffer("sign_unknown", unknown)
        self.sign_logit = nn.Parameter(torch.zeros(self.N))
        if bool(unknown.any()):
            u = unknown.numpy()[Wn.col[order]] > 0
            with torch.no_grad():
                self.w_val[torch.from_numpy(u)] = self.w_val[torch.from_numpy(u)].abs()

        self.gain = nn.Parameter(torch.ones(self.N))
        self.bias = nn.Parameter(torch.zeros(self.N))
        self.leak_logit = nn.Parameter(torch.zeros(self.N))

        if self.cfg.neuron_descriptors:
            D = self.cfg.neuron_descriptors
            self.eta = nn.Parameter(torch.randn(self.N, D) * 0.05)
            self.neuron_mlp = nn.Sequential(
                nn.Linear(1 + D, 32), nn.SiLU(), nn.Linear(32, 1))
        else:
            self.eta = None

        # ---- sensory ports -------------------------------------------------
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

        # Each real sensory population receives only the physical variables its
        # modality can plausibly measure. This is both smaller than the previous
        # all-proprio MLP and prevents next-gate coordinates leaking into the actor.
        self.sensory_encoders = nn.ModuleDict()
        for name, feat in SENSOR_FEATURES.items():
            rows_np = circuit.afferent.get(name, np.array([], dtype=np.int64))
            if not len(rows_np):
                continue
            if max(feat) >= proprio_dim:
                raise ValueError(f"sensor feature map for {name} exceeds proprio_dim={proprio_dim}")
            rows = torch.from_numpy(rows_np)
            features = torch.tensor(feat, dtype=torch.long)
            self.register_buffer(f"sensor_rows_{name}", rows)
            self.register_buffer(f"sensor_feat_{name}", features)
            hidden = min(self.cfg.sensory_hidden, max(8, 2 * len(feat)))
            encoder = nn.Sequential(
                nn.Linear(len(feat), hidden), nn.SiLU(), nn.Linear(hidden, len(rows_np))
            )
            nn.init.zeros_(encoder[-1].bias)
            nn.init.normal_(encoder[-1].weight, std=0.05)
            self.sensory_encoders[name] = encoder

        # ---- motor readout --------------------------------------------------
        self.motor = MotorDecoder(circuit, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), self.cfg.log_std_init))

        # The critic may use privileged task geometry to reduce variance; the
        # policy actor cannot access it except through the biological sensors.
        self.value = nn.Sequential(
            nn.Linear(proprio_dim + 2 * len(MOTOR_GROUPS), self.cfg.value_hidden), nn.SiLU(),
            nn.Linear(self.cfg.value_hidden, self.cfg.value_hidden), nn.SiLU(),
            nn.Linear(self.cfg.value_hidden, 1))

        self.to(device)
        self.device = device

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(self.N, batch, device=self.device)

    def _W(self) -> torch.Tensor:
        val = self.w_val
        if self.edge_gain is not None:
            val = val * torch.exp(self.edge_gain.clamp(-2.0, 2.0))
        return torch.sparse_coo_tensor(self.w_idx, val, (self.N, self.N),
                                       is_coalesced=True)

    def presyn_sign(self) -> torch.Tensor:
        return (1.0 - self.sign_unknown
                + self.sign_unknown * torch.tanh(self.sign_logit)).unsqueeze(1)

    def forward(self, obs: dict, state: torch.Tensor):
        proprio = obs["proprio"]
        B = proprio.shape[0]
        drive = torch.zeros(self.N, B, device=proprio.device)

        if "on" in obs and len(self._ret):
            on = obs["on"][:, self.ret_col].T
            off = obs["off"][:, self.ret_col].T
            rd = self.ret_is_on.unsqueeze(1) * on + (1 - self.ret_is_on).unsqueeze(1) * off
            drive.index_add_(0, self._ret, rd * self.retina_gain)

        for name, encoder in self.sensory_encoders.items():
            rows = getattr(self, f"sensor_rows_{name}")
            feat = getattr(self, f"sensor_feat_{name}")
            drive.index_add_(0, rows, encoder(proprio.index_select(1, feat)).T)

        W = self._W()
        sign = self.presyn_sign()
        lam = torch.sigmoid(self.leak_logit).unsqueeze(1)
        h = state * self.cfg.carry

        for _ in range(self.cfg.n_iters):
            m = torch.sparse.mm(W, h * sign) + drive
            z = self.gain.unsqueeze(1) * m + self.bias.unsqueeze(1)
            if self.eta is not None:
                z = z + self.neuron_mlp(
                    torch.cat([m.unsqueeze(-1),
                               self.eta.unsqueeze(1).expand(-1, B, -1)], -1)).squeeze(-1)
            h = (1 - lam) * h + lam * self.phi(z)

        mean, group_feat = self.motor(h)
        value = self.value(torch.cat([proprio, group_feat], -1)).squeeze(-1)
        return mean, self.log_std.expand_as(mean), value, h

    def dist(self, mean, log_std):
        return torch.distributions.Normal(mean, log_std.exp())

    def act(self, obs, state, deterministic: bool = False):
        mean, log_std, value, h = self.forward(obs, state)
        if deterministic:
            return torch.tanh(mean), None, value, h
        d = self.dist(mean, log_std)
        raw = d.rsample()
        return torch.tanh(raw), squashed_log_prob(d, raw), value, h, raw


class MotorDecoder(nn.Module):
    """Efferent activity -> the 15 wing controls through anatomical motor groups."""

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
        self.group_rows = rows
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
                if group not in gi or ctrl not in ci:
                    continue
                col = 2 * gi[group] + (0 if mode == "sum" else 1)
                self.readout.weight[ci[ctrl], col] = w

        self.all_rows = torch.cat(rows) if rows else torch.zeros(0, dtype=torch.long)
        self.register_buffer("_all", self.all_rows)
        self.residual = nn.Linear(max(len(self.all_rows), 1), action_dim)
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
        out = self.readout(gf) + self.residual(h[self._all].T)
        return out, gf
