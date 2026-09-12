"""Controls for the experiment.

The interesting claim is not "a network can learn to fly" — that is well
established. It is that *this particular* wiring diagram helps. That claim is
only testable against matched controls, so they ship in the box:

  * ``shuffle_connectome(mode="degree")`` — rewires the graph while preserving
    every neuron's in- and out-degree. Same number of neurons, same number of
    synapses, same degree sequence, same E/I ratio. Only the *pattern* of who
    connects to whom is destroyed. This is the control that matters.
  * ``shuffle_connectome(mode="erdos")`` — a random graph matched on node and
    edge count only.
  * ``MLPPolicy`` — a conventional network with a comparable parameter budget.

If the real connectome does not beat degree-preserving shuffle, the connectome
is doing nothing and you have learned something worth knowing.
"""
from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from ..connectome.circuits import FlightCircuit
from .connectome_policy import squashed_log_prob


def shuffle_connectome(circuit: FlightCircuit, mode: str = "degree",
                       seed: int = 0) -> FlightCircuit:
    """Return a copy of the circuit with its edges randomised."""
    rng = np.random.default_rng(seed)
    W = circuit.cx.W.tocoo()
    n = W.shape[0]

    if mode == "erdos":
        rows = rng.integers(0, n, W.nnz)
        cols = rng.integers(0, n, W.nnz)
        data = rng.permutation(W.data)
    elif mode == "degree":
        # Preserve the degree sequence exactly by permuting the endpoint lists
        # independently — the configuration model.
        rows = rng.permutation(W.row)
        cols = rng.permutation(W.col)
        data = W.data.copy()
    else:
        raise ValueError(mode)

    Ws = sp.csr_matrix((data, (rows, cols)), shape=W.shape, dtype=np.float32)
    Ws.sum_duplicates()

    new = copy.copy(circuit)
    new.cx = copy.copy(circuit.cx)
    new.cx.W = Ws
    return new


class MLPPolicy(nn.Module):
    """Plain feedforward baseline over the same observations."""

    def __init__(self, proprio_dim: int, vision_dim: int, action_dim: int,
                 hidden: int = 512, depth: int = 3, vision_compress: int = 64,
                 log_std_init: float = -1.6, device: str = "cpu"):
        super().__init__()
        self.vision_dim = vision_dim
        self.vis = (nn.Sequential(nn.Linear(2 * vision_dim, vision_compress), nn.SiLU())
                    if vision_dim else None)
        d_in = proprio_dim + (vision_compress if vision_dim else 0)
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(d, action_dim)
        nn.init.normal_(self.mean.weight, std=0.01); nn.init.zeros_(self.mean.bias)
        self.v = nn.Linear(d, 1)
        # matched to PolicyConfig.log_std_init so the baseline explores with the
        # same action noise as the connectome policy; otherwise the comparison
        # measures initial exploration scale rather than architecture
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        self.to(device)
        self.device = device

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(self, batch: int):
        return torch.zeros(1, batch, device=self.device)

    def forward(self, obs, state):
        x = obs["proprio"]
        if self.vis is not None and "on" in obs:
            x = torch.cat([x, self.vis(torch.cat([obs["on"], obs["off"]], -1))], -1)
        z = self.trunk(x)
        return self.mean(z), self.log_std.expand(x.shape[0], -1), self.v(z).squeeze(-1), state

    def dist(self, mean, log_std):
        return torch.distributions.Normal(mean, log_std.exp())

    def act(self, obs, state, deterministic: bool = False):
        mean, log_std, value, h = self.forward(obs, state)
        if deterministic:
            return torch.tanh(mean), None, value, h
        d = self.dist(mean, log_std)
        raw = d.rsample()
        return torch.tanh(raw), squashed_log_prob(d, raw), value, h, raw
