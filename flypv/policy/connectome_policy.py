"""The fly's own wiring diagram, used as the policy network.

The whole point is that **the connectivity is not learned**. `W` comes from
28,361 real neurons and 534,681 real measured connections, with the sign of
every synapse set by the presynaptic neuron's predicted neurotransmitter. It is
registered as a buffer, not a parameter. Gradient descent is only allowed to
tune what a neuron *is* — its gain, its threshold, its membrane time constant —
not who it talks to. That is the same freedom a real fly's development and
neuromodulation have.

Dynamics are a leaky rate network, the standard abstraction for connectome-
constrained modelling:

    m_t  = W_norm @ h_{t-1}  +  I_t            (synaptic input + sensory drive)
    h_t  = (1 - lam_v) h_{t-1} + lam_v * phi(a_v * m_t + b_v)

with `a_v`, `b_v`, `lam_v` learnable per neuron. `K` of these iterations run per
control step: information physically has to cross several synapses to get from
an eye to a wing, and K sets how many it may cross within one wingbeat.

Row-normalising `W` by each neuron's total input strength is what keeps a hub
neuron with 10,000 inputs from saturating instantly, and it preserves the
*relative* weighting between a neuron's partners — which is the part the
connectome actually measured.
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
    n_iters: int = 4            # synaptic hops allowed per control step
    carry: float = 0.5          # how much neural state persists between steps
    learn_synapses: bool = False  # if True, a per-edge gain is trainable too
    neuron_descriptors: int = 0   # >0 adds FlyGM-style per-neuron descriptor MLP
    act: str = "softplus"       # firing rates are non-negative
    value_hidden: int = 256
    log_std_init: float = -1.6
    sensory_hidden: int = 64


def squashed_log_prob(dist, raw: torch.Tensor) -> torch.Tensor:
    """log pi(tanh(raw)) — the Gaussian density with the tanh Jacobian removed.

    This correction has to be applied identically when the action is sampled and
    when its likelihood is recomputed during the PPO update. Applying it in only
    one place leaves a large offset in every importance ratio, which shows up as
    a KL in the single digits instead of the hundredths.
    """
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
        # row-normalise by total input strength: relative synapse counts are what
        # the connectome measured, absolute counts vary 1000-fold across cells
        row_abs = np.asarray(np.abs(W).sum(axis=1)).ravel()
        scale = 1.0 / np.maximum(row_abs, 1.0)
        Wn = sp.diags(scale.astype(np.float32)) @ W
        Wn = Wn.tocoo()

        order = np.lexsort((Wn.col, Wn.row))       # coalesce once, at build time
        idx = torch.from_numpy(np.stack([Wn.row[order], Wn.col[order]]).astype(np.int64))
        val = torch.from_numpy(Wn.data[order].astype(np.float32))
        self.register_buffer("w_idx", idx)
        self.register_buffer("w_val", val)
        self.n_edges = val.numel()

        if self.cfg.learn_synapses:
            # a bounded per-edge gain: the wiring is fixed, its efficacy is not
            self.edge_gain = nn.Parameter(torch.zeros(self.n_edges))
        else:
            self.edge_gain = None

        # 2.5% of neurons have no confident transmitter call. Give those a
        # learnable sign — but apply it to the *presynaptic neuron*, not to its
        # edges. A synapse's sign is a property of the cell that makes it, so
        # W_signed @ h is identical to W_magnitude @ (sign * h), and the second
        # form keeps the gradient dense and per-neuron (N values) instead of
        # per-edge and sparse (534k values). That one change is the difference
        # between a backward pass 50x the cost of the forward and one at parity.
        unknown = torch.from_numpy((circuit.cx.nt_sign == 0).astype(np.float32))
        self.register_buffer("sign_unknown", unknown)
        self.sign_logit = nn.Parameter(torch.zeros(self.N))
        if bool(unknown.any()):
            # store magnitudes for the unlabelled presynaptic neurons; their sign
            # is reapplied at run time through `presyn_sign()`
            u = unknown.numpy()[Wn.col[order]] > 0
            v = self.w_val.numpy() if hasattr(self.w_val, "numpy") else None
            with torch.no_grad():
                self.w_val[torch.from_numpy(u)] = self.w_val[torch.from_numpy(u)].abs()

        # per-neuron intrinsic properties — the only "neuron-level" freedom
        self.gain = nn.Parameter(torch.ones(self.N))
        self.bias = nn.Parameter(torch.zeros(self.N))
        self.leak_logit = nn.Parameter(torch.zeros(self.N))   # sigmoid -> lam in (0,1)

        if self.cfg.neuron_descriptors:
            D = self.cfg.neuron_descriptors
            self.eta = nn.Parameter(torch.randn(self.N, D) * 0.05)
            self.neuron_mlp = nn.Sequential(
                nn.Linear(1 + D, 32), nn.SiLU(), nn.Linear(32, 1))
        else:
            self.eta = None

        # ---- sensory ports -------------------------------------------------
        self.retina_rows = torch.from_numpy(circuit.afferent.get("retina", np.array([], np.int64)))
        other = [v for k, v in circuit.afferent.items() if k != "retina" and len(v)]
        self.other_rows = torch.from_numpy(
            np.concatenate(other) if other else np.array([], np.int64))
        self.register_buffer("_ret", self.retina_rows)
        self.register_buffer("_oth", self.other_rows)

        # Which ommatidial column does each retinotopic neuron belong to, and is
        # it on the ON or the OFF side of the first synapse? R1-R6 are
        # histaminergic and inhibit the lamina, so L1/L5/Mi1 report contrast
        # increments and L2/L3/Tm1/Tm2/Tm9 report decrements.
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

        h = self.cfg.sensory_hidden
        self.sensory_encoder = nn.Sequential(
            nn.Linear(proprio_dim, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.sensory_to_neurons = nn.Linear(h, max(len(self.other_rows), 1))
        nn.init.zeros_(self.sensory_to_neurons.bias)
        nn.init.normal_(self.sensory_to_neurons.weight, std=0.05)
        self.retina_gain = nn.Parameter(torch.tensor(1.0))

        # ---- motor readout --------------------------------------------------
        self.motor = MotorDecoder(circuit, action_dim)

        self.log_std = nn.Parameter(torch.full((action_dim,), self.cfg.log_std_init))

        # value head sees the same afferent summary plus pooled neural state
        self.value = nn.Sequential(
            nn.Linear(proprio_dim + 2 * len(MOTOR_GROUPS), self.cfg.value_hidden), nn.SiLU(),
            nn.Linear(self.cfg.value_hidden, self.cfg.value_hidden), nn.SiLU(),
            nn.Linear(self.cfg.value_hidden, 1))

        self.to(device)
        self.device = device

    # ------------------------------------------------------------------
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(self.N, batch, device=self.device)

    def _W(self) -> torch.Tensor:
        val = self.w_val
        if self.edge_gain is not None:
            # NOTE: this makes the sparse *values* trainable, and differentiating
            # through them costs far more than the fixed-structure path. Off by
            # default; the connectome is meant to be a constraint, not a warm start.
            val = val * torch.exp(self.edge_gain.clamp(-2.0, 2.0))
        return torch.sparse_coo_tensor(self.w_idx, val, (self.N, self.N),
                                       is_coalesced=True)

    def presyn_sign(self) -> torch.Tensor:
        """(N, 1) multiplier carrying the learnable sign of unlabelled neurons."""
        return (1.0 - self.sign_unknown
                + self.sign_unknown * torch.tanh(self.sign_logit)).unsqueeze(1)

    # ------------------------------------------------------------------
    def forward(self, obs: dict, state: torch.Tensor):
        """obs: proprio (B,P), on/off (B,C). state: (N,B). Returns dist params, value, state."""
        proprio = obs["proprio"]
        B = proprio.shape[0]

        drive = torch.zeros(self.N, B, device=proprio.device)

        # vision lands on the exact neurons that see it
        if "on" in obs and len(self._ret):
            on = obs["on"][:, self.ret_col].T                 # (n_ret, B)
            off = obs["off"][:, self.ret_col].T
            rd = self.ret_is_on.unsqueeze(1) * on + (1 - self.ret_is_on).unsqueeze(1) * off
            drive.index_add_(0, self._ret, rd * self.retina_gain)

        # the non-visual sensors are encoded and routed to their own afferents
        if len(self._oth):
            enc = self.sensory_encoder(proprio)
            drive.index_add_(0, self._oth, self.sensory_to_neurons(enc).T)

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
    """Efferent activity -> the 15 wing controls, wired the way the muscles are.

    Rather than flattening every motor neuron into one opaque linear layer, we
    pool each motor group per side — `power`, `basalar`, `hinge`, and so on, left
    and right — because that is the functional unit: a steering muscle group on
    one side is one knob on the wing hinge. The map from those pooled features to
    the control vector is initialised to the known anatomy (power muscles drive
    amplitude and frequency, hinge muscles drive angle of attack, and a
    left-right *difference* drives roll and yaw), then refined by training.

    A low-gain residual from the full efferent population is added on top, so the
    model can express something the coarse grouping misses.
    """

    #: (group, mode) -> control index, mode is "sum" (bilateral) or "diff" (asymmetric)
    WIRING: list[tuple[str, str, str, float]] = [
        ("power",     "sum",  "freq", 1.0),
        ("power",     "sum",  "amp_R", 0.7), ("power", "sum", "amp_L", 0.7),
        ("power",     "diff", "amp_R", 0.6), ("power", "diff", "amp_L", -0.6),
        ("basalar",   "sum",  "amp_R", 0.8), ("basalar", "sum", "amp_L", 0.8),
        ("basalar",   "diff", "amp_R", 1.0), ("basalar", "diff", "amp_L", -1.0),
        ("pterale",   "sum",  "offset_R", 0.9), ("pterale", "sum", "offset_L", 0.9),
        ("pterale",   "diff", "offset_R", 0.8), ("pterale", "diff", "offset_L", -0.8),
        ("axillary",  "sum",  "stroke_tilt", 1.0),
        ("axillary",  "diff", "dev_R", 0.7), ("axillary", "diff", "dev_L", -0.7),
        ("hinge",     "sum",  "aoa_down_R", 1.0), ("hinge", "sum", "aoa_down_L", 1.0),
        ("hinge",     "diff", "aoa_down_R", 0.9), ("hinge", "diff", "aoa_down_L", -0.9),
        ("wing_misc", "sum",  "aoa_up_R", 0.8), ("wing_misc", "sum", "aoa_up_L", 0.8),
        ("wing_misc", "diff", "aoa_up_R", 0.8), ("wing_misc", "diff", "aoa_up_L", -0.8),
        ("neck",      "sum",  "head_pitch", 1.0),
        ("neck",      "diff", "head_yaw", 1.0),
        ("abdomen",   "sum",  "abdomen_pitch", 1.0),
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
            a = h[rows]                                  # (n_g, B)
            feats.append(a.mean(0))                      # bilateral sum -> symmetric drive
            denom = side.abs().sum().clamp(min=1.0)
            feats.append((a * side).sum(0) / denom)      # left-right difference -> steering
        gf = torch.stack(feats, -1)                      # (B, 2G)
        out = self.readout(gf) + self.residual(h[self._all].T)
        return out, gf
