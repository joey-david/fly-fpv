"""Sparse neural network whose mask is the Drosophila flight connectome.

The connectome is an architectural constraint, not a physiological simulator.
Each measured neuron is a unit and each measured directed connection is an
allowed weight. Edge values, signs, node biases and input/output adapters are
learned from the flight task. Every control decision is feed-forward: the same
connectome mask is unrolled for a small number of residual sparse layers, with
no hidden state carried between simulator steps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..connectome.circuits import FlightCircuit

WING_MOTOR_GROUPS = ("power", "basalar", "axillary", "hinge", "pterale", "wing_misc")


@dataclass
class PolicyConfig:
    n_iters: int = 6
    learn_synapses: bool = True
    act: str = "silu"
    value_hidden: int = 256
    log_std_init: float = -1.6
    sensory_hidden: int = 64
    edge_init_scale: float = 0.7


def squashed_log_prob(dist, raw: torch.Tensor) -> torch.Tensor:
    logp = dist.log_prob(raw).sum(-1)
    return logp - (2 * (math.log(2) - raw - F.softplus(-2 * raw))).sum(-1)


def _act(name: str):
    return {
        "silu": F.silu,
        "gelu": F.gelu,
        "tanh": torch.tanh,
        "relu": F.relu,
        "elu": F.elu,
    }[name]


class ConnectomePolicy(nn.Module):
    """Trainable sparse network constrained only by the measured graph mask.

    The directed MaleCNS adjacency is repeated for ``n_iters`` feed-forward
    residual layers. Every allowed edge has an independent trainable weight in
    every layer; absent connectome edges stay identically zero. Biological
    synapse counts and transmitter signs are deliberately *not* constraints.

    Visual ON/OFF signals enter retinotopically. The remaining low-dimensional
    task/proprio vector is learned into non-visual afferents. Actions are decoded
    only from anatomically identified efferent neurons, but their mapping to the
    15 simulator controls is fully learned.
    """

    def __init__(self, circuit: FlightCircuit, proprio_dim: int, action_dim: int,
                 cfg: PolicyConfig | None = None, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        self.circuit = circuit
        self.N = circuit.n
        self.action_dim = action_dim
        self.phi = _act(self.cfg.act)

        # W[post, pre]. Only row/column coordinates survive: measured magnitudes
        # and neurotransmitter signs are not part of the optimization constraint.
        coo = circuit.cx.W.tocoo()
        order = np.lexsort((coo.col, coo.row))
        rows = coo.row[order].astype(np.int64, copy=False)
        cols = coo.col[order].astype(np.int64, copy=False)
        edge_index = torch.from_numpy(np.stack([rows, cols]))
        self.register_buffer("edge_index", edge_index)
        self.n_edges = len(rows)

        # Fan-in-scaled random initialization, evaluated only on allowed edges.
        # Each unrolled layer gets independent values; topology is shared.
        indegree = np.bincount(rows, minlength=self.N).astype(np.float32)
        edge_std = self.cfg.edge_init_scale / np.sqrt(np.maximum(indegree[rows], 1.0))
        init = torch.randn(self.cfg.n_iters, self.n_edges)
        init *= torch.from_numpy(edge_std).unsqueeze(0)
        if self.cfg.learn_synapses:
            self.edge_weight = nn.Parameter(init)
        else:
            self.register_buffer("edge_weight", init)

        self.node_bias = nn.Parameter(torch.zeros(self.cfg.n_iters, self.N))
        self.residual_logit = nn.Parameter(torch.zeros(self.cfg.n_iters))

        # Vision keeps only the useful anatomical constraint: retinotopy.
        self.retina_rows = torch.from_numpy(
            circuit.afferent.get("retina", np.array([], dtype=np.int64))
        )
        self.register_buffer("_ret", self.retina_rows)
        cols_meta = sorted({v for v in circuit.hex_coords.values()})
        col_ix = {c: i for i, c in enumerate(cols_meta)}
        ret_col = []
        for r in self.retina_rows.tolist():
            key = circuit.hex_coords.get(int(r))
            ret_col.append(col_ix.get(key, 0) if key is not None else 0)
        self.register_buffer("ret_col", torch.tensor(ret_col, dtype=torch.long))
        self.n_columns = len(cols_meta)

        # Each retinotopic neuron learns its own mixture of local ON/OFF.
        self.retina_mix = nn.Parameter(torch.empty(len(self.retina_rows), 2))
        nn.init.normal_(self.retina_mix, std=1.0 / math.sqrt(2.0))
        self.retina_bias = nn.Parameter(torch.zeros(len(self.retina_rows)))

        # All other task/proprio features get a learned adapter into the
        # non-retinal sensory population. We do not impose modality semantics.
        other = [v for k, v in circuit.afferent.items() if k != "retina" and len(v)]
        other_rows = np.unique(np.concatenate(other)) if other else np.array([], dtype=np.int64)
        self.register_buffer("_oth", torch.from_numpy(other_rows))
        if len(other_rows):
            h = self.cfg.sensory_hidden
            self.sensory_encoder = nn.Sequential(
                nn.Linear(proprio_dim, h), nn.SiLU(),
                nn.Linear(h, len(other_rows)),
            )
            nn.init.normal_(self.sensory_encoder[-1].weight, std=0.02)
            nn.init.zeros_(self.sensory_encoder[-1].bias)
        else:
            self.sensory_encoder = None

        # Outputs are anatomically anchored, but not hand-wired. Wing controls
        # can use any flight motor neuron; head and abdomen use their own motor
        # populations. Within those families the readout is fully learned.
        motor_rows = circuit.efferent_idx
        if not len(motor_rows):
            raise ValueError("connectome policy requires at least one efferent neuron")
        self.register_buffer("motor_rows", torch.from_numpy(motor_rows))

        wing_parts = [circuit.efferent.get(k, np.array([], dtype=np.int64))
                      for k in WING_MOTOR_GROUPS]
        wing_rows = np.unique(np.concatenate([x for x in wing_parts if len(x)]))
        head_rows = np.unique(circuit.efferent.get("neck", np.array([], dtype=np.int64)))
        abdomen_rows = np.unique(circuit.efferent.get("abdomen", np.array([], dtype=np.int64)))
        if not len(wing_rows):
            raise ValueError("connectome policy requires flight motor neurons")
        self.register_buffer("wing_motor_rows", torch.from_numpy(wing_rows))
        self.register_buffer("head_motor_rows", torch.from_numpy(head_rows))
        self.register_buffer("abdomen_motor_rows", torch.from_numpy(abdomen_rows))

        self.wing_readout = nn.Linear(len(wing_rows), min(action_dim, 12))
        self.head_readout = nn.Linear(max(len(head_rows), 1), 2)
        self.abdomen_readout = nn.Linear(max(len(abdomen_rows), 1), 1)
        for readout in (self.wing_readout, self.head_readout, self.abdomen_readout):
            nn.init.normal_(readout.weight, std=0.02)
            nn.init.zeros_(readout.bias)

        # The critic is a training aid, so it is intentionally unconstrained.
        self.value = nn.Sequential(
            nn.Linear(proprio_dim + len(motor_rows), self.cfg.value_hidden), nn.SiLU(),
            nn.Linear(self.cfg.value_hidden, self.cfg.value_hidden), nn.SiLU(),
            nn.Linear(self.cfg.value_hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), self.cfg.log_std_init))

        self.to(device)
        self.device = device

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(self, batch: int) -> torch.Tensor:
        """Compatibility helper; the policy never consumes previous state."""
        return torch.zeros(self.N, batch, device=self.device)

    def _sparse_weight(self, layer: int) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self.edge_index,
            self.edge_weight[layer],
            (self.N, self.N),
            is_coalesced=True,
            check_invariants=False,
        )

    def _input_drive(self, obs: dict) -> torch.Tensor:
        proprio = obs["proprio"]
        B = proprio.shape[0]
        drive = torch.zeros(self.N, B, device=proprio.device, dtype=proprio.dtype)

        if "on" in obs and len(self._ret):
            on = obs["on"][:, self.ret_col].T
            off = obs["off"][:, self.ret_col].T
            rd = (self.retina_mix[:, 0:1] * on
                  + self.retina_mix[:, 1:2] * off
                  + self.retina_bias[:, None])
            drive.index_add_(0, self._ret, rd)

        if self.sensory_encoder is not None and len(self._oth):
            drive.index_add_(0, self._oth, self.sensory_encoder(proprio).T)
        return drive

    def forward(self, obs: dict, state: torch.Tensor | None = None):
        proprio = obs["proprio"]
        drive = self._input_drive(obs)
        h = self.phi(drive)

        # Deep sparse residual network. No activity crosses simulator timesteps;
        # n_iters is simply feed-forward depth on a repeated connectome mask.
        for layer in range(self.cfg.n_iters):
            m = torch.sparse.mm(self._sparse_weight(layer), h)
            z = m + self.node_bias[layer].unsqueeze(1)
            a = torch.sigmoid(self.residual_logit[layer])
            h = (h + a * self.phi(z)) / torch.sqrt(1.0 + a * a)

        motor = h.index_select(0, self.motor_rows).T
        wing = h.index_select(0, self.wing_motor_rows).T
        parts = [self.wing_readout(wing)]
        if self.action_dim > 12:
            if len(self.head_motor_rows):
                head = h.index_select(0, self.head_motor_rows).T
            else:
                head = torch.zeros(wing.shape[0], 1, device=h.device, dtype=h.dtype)
            parts.append(self.head_readout(head))
        if self.action_dim > 14:
            if len(self.abdomen_motor_rows):
                abdomen = h.index_select(0, self.abdomen_motor_rows).T
            else:
                abdomen = torch.zeros(wing.shape[0], 1, device=h.device, dtype=h.dtype)
            parts.append(self.abdomen_readout(abdomen))
        mean = torch.cat(parts, -1)[..., :self.action_dim]
        value = self.value(torch.cat([proprio, motor], -1)).squeeze(-1)
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
