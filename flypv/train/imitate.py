"""Behaviour cloning from the reflex pilot, before PPO takes over.

Flapping hover is an unstable equilibrium, so a freshly initialised policy
crashes within a few control steps and almost every rollout it collects is the
same short failure. There is nothing for the advantage estimator to compare.
Cloning a competent-but-mediocre controller first moves the policy into the part
of state space where the task's reward structure is actually informative, and
PPO then improves on it — this is the two-stage recipe connectome-policy work
uses, and the reason the cloning target is deliberately *not* very good is that
we want PPO to have somewhere to go.

The loss is on the squashed action rather than the pre-tanh sample, so the
saturated controls the expert produces near the limits do not blow up into
infinite targets.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..control import ReflexPilot
from ..monitor.bus import BUS


def collect(env, pilot: ReflexPilot, steps: int, device: str):
    """Run the expert and record (observation, action) pairs."""
    obs = env.reset()
    P, A = [], []
    for _ in range(steps):
        a = pilot(env)
        P.append({k: v.copy() for k, v in obs.items()})
        A.append(a.copy())
        obs, *_ = env.step(a)
    out = {k: torch.from_numpy(np.concatenate([p[k] for p in P])).to(device)
           for k in P[0]}
    return out, torch.from_numpy(np.concatenate(A)).float().to(device)


def warm_start(policy, env, steps: int = 400, epochs: int = 8, batch: int = 1024,
               lr: float = 1e-3, noise: float = 0.06, verbose: bool = True):
    """Clone the reflex pilot into `policy`. Returns the final mean error."""
    device = policy.device
    pilot = ReflexPilot(noise=noise)
    if verbose:
        print(f"[warmstart] collecting {steps * env.cfg.n_envs:,} expert transitions ...")
    obs, expert = collect(env, pilot, steps, device)

    n = expert.shape[0]
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    idx = np.arange(n)
    err = float("nan")
    for ep in range(epochs):
        np.random.shuffle(idx)
        losses = []
        for s in range(0, n, batch):
            j = torch.from_numpy(idx[s : s + batch]).to(device)
            o = {k: v[j] for k, v in obs.items()}
            mean, _, value, _ = policy.forward(o, policy.init_state(len(j)))
            loss = F.mse_loss(torch.tanh(mean), expert[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        err = float(np.mean(losses))
        BUS.record(warmstart_epoch=ep + 1, warmstart_loss=err)
        if verbose:
            print(f"[warmstart] epoch {ep + 1}/{epochs}  action MSE {err:.5f}")
    return err
