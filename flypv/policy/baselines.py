"""Matched controls for the connectome experiment."""
from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from ..connectome.circuits import FlightCircuit
from .connectome_policy import squashed_log_prob


def _degree_preserving_rewire(
    W: sp.spmatrix,
    rng: np.random.Generator,
    swaps_per_edge: float = 2.0,
) -> sp.csr_matrix:
    """Directed double-edge swaps preserving every node's in/out degree."""
    coo = W.tocoo(copy=True)
    rows = coo.row.astype(np.int64, copy=True)
    cols = coo.col.astype(np.int64, copy=True)
    data = coo.data.astype(np.float32, copy=True)
    m = len(data)
    edges = {(int(r), int(c)) for r, c in zip(rows, cols)}
    if len(edges) != m:
        raise ValueError("degree-preserving shuffle requires a coalesced sparse matrix")

    target = int(swaps_per_edge * m)
    accepted = 0
    attempts = 0
    max_attempts = max(target * 20, 1000)
    while accepted < target and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        r1, c1 = int(rows[i]), int(cols[i])
        r2, c2 = int(rows[j]), int(cols[j])
        if r1 == r2 or c1 == c2:
            continue
        e1 = (r2, c1)
        e2 = (r1, c2)
        if e1[0] == e1[1] or e2[0] == e2[1]:
            continue
        old1, old2 = (r1, c1), (r2, c2)
        if (e1 in edges and e1 not in (old1, old2)) or (e2 in edges and e2 not in (old1, old2)):
            continue

        edges.remove(old1)
        edges.remove(old2)
        edges.add(e1)
        edges.add(e2)
        rows[i], rows[j] = r2, r1
        accepted += 1

    if accepted < target:
        print(f"[baseline] degree shuffle accepted {accepted:,}/{target:,} requested edge swaps")

    out = sp.csr_matrix((data, (rows, cols)), shape=W.shape, dtype=np.float32)
    out.sum_duplicates()
    if out.nnz != W.nnz:
        raise RuntimeError("degree-preserving rewire changed edge count")
    return out


def _erdos_exact(W: sp.spmatrix, rng: np.random.Generator) -> sp.csr_matrix:
    """Random directed graph with exactly the same node and edge counts."""
    n = W.shape[0]
    m = W.nnz
    flat: set[int] = set()
    while len(flat) < m:
        need = m - len(flat)
        batch = rng.integers(0, n * n, size=max(4096, int(need * 1.05)))
        flat.update(int(x) for x in batch)
    ids = np.fromiter(flat, dtype=np.int64, count=m)
    rows, cols = ids // n, ids % n
    data = rng.permutation(W.data).astype(np.float32, copy=False)
    out = sp.csr_matrix((data, (rows, cols)), shape=W.shape, dtype=np.float32)
    if out.nnz != m:
        raise RuntimeError("Erdos baseline failed to preserve edge count")
    return out


def shuffle_connectome(
    circuit: FlightCircuit,
    mode: str = "degree",
    seed: int = 0,
) -> FlightCircuit:
    """Return a copy of the circuit with only its adjacency control changed."""
    rng = np.random.default_rng(seed)
    W = circuit.cx.W.tocsr()
    if mode == "degree":
        shuffled = _degree_preserving_rewire(W, rng)
    elif mode == "erdos":
        shuffled = _erdos_exact(W, rng)
    else:
        raise ValueError(mode)

    new = copy.copy(circuit)
    new.cx = copy.copy(circuit.cx)
    new.cx.W = shuffled
    return new


class MLPPolicy(nn.Module):
    """Dense feed-forward control with a similar parameter budget.

    A six-layer flight connectome has roughly 3--4M trainable parameters. The
    1280-wide default keeps the dense baseline in that regime so topology
    comparisons are not just capacity comparisons.
    """

    def __init__(
        self,
        proprio_dim: int,
        vision_dim: int,
        action_dim: int,
        hidden: int = 1280,
        depth: int = 3,
        vision_compress: int = 32,
        log_std_init: float = -1.6,
        device: str = "cpu",
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.vis = (
            nn.Sequential(nn.Linear(2 * vision_dim, vision_compress), nn.SiLU())
            if vision_dim
            else None
        )
        d_in = proprio_dim + (vision_compress if vision_dim else 0)
        layers = []
        d = d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(d, action_dim)
        nn.init.normal_(self.mean.weight, std=0.01)
        nn.init.zeros_(self.mean.bias)
        self.v = nn.Linear(d, 1)
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
        dist = self.dist(mean, log_std)
        raw = dist.rsample()
        return torch.tanh(raw), squashed_log_prob(dist, raw), value, h, raw
