"""Behaviour cloning from the reflex pilot before PPO takes over.

The sparse connectome network is intentionally initialized as a generic neural
network: only its mask is anatomical. A stronger cloning stage therefore gives
PPO a competent starting controller instead of asking RL to discover basic
stability and racing simultaneously.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..control import ReflexPilot
from ..monitor.bus import BUS


def collect(env, pilot: ReflexPilot, steps: int):
    """Run the expert and keep the imitation dataset on CPU."""
    obs = env.reset()
    P, A = [], []
    for _ in range(steps):
        a = pilot(env)
        P.append({k: v.copy() for k, v in obs.items()})
        A.append(a.copy())
        obs, *_ = env.step(a)
    out = {
        k: torch.from_numpy(np.concatenate([p[k] for p in P])).float()
        for k in P[0]
    }
    expert = torch.from_numpy(np.concatenate(A)).float()
    return out, expert


def _episode_stats(env, action_fn, target_episodes: int) -> dict[str, float]:
    obs = env.reset()
    gates: list[float] = []
    crashes: list[float] = []
    returns: list[float] = []
    max_steps = env.max_steps * max(2, math.ceil(target_episodes / env.cfg.n_envs) + 1)
    for _ in range(max_steps):
        action = action_fn(obs)
        obs, _, term, trunc, info = env.step(action)
        done = term | trunc
        ids = np.flatnonzero(done)
        if len(ids):
            gates.extend(info["gates_done"][ids].astype(float).tolist())
            crashes.extend(info["crashed"][ids].astype(float).tolist())
            returns.extend(info["episode_return"][ids].astype(float).tolist())
        if len(gates) >= target_episodes:
            break
    if not gates:
        return {"gates": 0.0, "crash": 1.0, "return": 0.0}
    n = min(target_episodes, len(gates))
    return {
        "gates": float(np.mean(gates[:n])),
        "crash": float(np.mean(crashes[:n])),
        "return": float(np.mean(returns[:n])),
    }


def evaluate_expert(env, episodes: int | None = None) -> dict[str, float]:
    pilot = ReflexPilot(noise=0.0)
    target = episodes or max(32, 2 * env.cfg.n_envs)
    return _episode_stats(env, lambda _obs: pilot(env), target)


def evaluate_policy(policy, env, episodes: int | None = None) -> dict[str, float]:
    target = episodes or max(32, 2 * env.cfg.n_envs)

    def act(obs):
        o = {k: torch.from_numpy(v).float().to(policy.device) for k, v in obs.items()}
        with torch.no_grad():
            action, _, _, _ = policy.act(o, deterministic=True)
        return action.cpu().numpy()

    return _episode_stats(env, act, target)


def warm_start(policy, env, steps: int = 800, epochs: int = 10, batch: int = 512,
               lr: float = 1e-3, noise: float = 0.06, verbose: bool = True):
    """Clone the reflex pilot and report actual pre-PPO flight quality."""
    device = policy.device

    if verbose:
        expert_stats = evaluate_expert(env)
        print(
            "[warmstart] expert  "
            f"gates {expert_stats['gates']:.2f}/{env.cfg.n_gates} | "
            f"crash {expert_stats['crash']:.2f} | ret {expert_stats['return']:.2f}"
        )

    pilot = ReflexPilot(noise=noise)
    if verbose:
        print(f"[warmstart] collecting {steps * env.cfg.n_envs:,} expert transitions ...")
    obs, expert = collect(env, pilot, steps)

    n = expert.shape[0]
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    idx = np.arange(n)
    err = float("nan")
    for ep in range(epochs):
        np.random.shuffle(idx)
        losses = []
        for s in range(0, n, batch):
            j_cpu = torch.from_numpy(idx[s:s + batch])
            o = {k: v.index_select(0, j_cpu).to(device) for k, v in obs.items()}
            target = expert.index_select(0, j_cpu).to(device)
            mean, _, _, _ = policy.forward(o)
            loss = F.mse_loss(torch.tanh(mean), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        err = float(np.mean(losses))
        BUS.record(warmstart_epoch=ep + 1, warmstart_loss=err)
        if verbose:
            print(f"[warmstart] epoch {ep + 1}/{epochs}  action MSE {err:.5f}")
        if err < 0.01:
            break

    if verbose:
        clone_stats = evaluate_policy(policy, env)
        print(
            "[warmstart] clone   "
            f"gates {clone_stats['gates']:.2f}/{env.cfg.n_gates} | "
            f"crash {clone_stats['crash']:.2f} | ret {clone_stats['return']:.2f}"
        )
        if clone_stats["gates"] < 0.5 and expert_stats["gates"] >= 0.5:
            print("[warmstart] warning: clone is still materially worse than the expert")
    return err
