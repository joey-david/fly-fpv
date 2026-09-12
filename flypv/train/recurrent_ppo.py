"""Persistent-state PPO with truncated backpropagation through the connectome.

The connectome is a recurrent dynamical system, so its neural state should not be
thrown away at every 5 ms control step. Rollouts therefore carry ``h`` across
steps and episode boundaries reset only the environments that actually ended.

Training replays the rollout in short temporal chunks. The state at each chunk
boundary is stored during collection, treated as a constant initial condition,
and gradients flow through the next ``tbptt_steps`` control steps before being
truncated. With the default 16-step chunk at 200 Hz this gives 80 ms of temporal
credit assignment while avoiding the cost of full-rollout BPTT over 28k neurons.

Set ``recurrent=False`` for the old stateless ablation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .ppo import PPO as _BasePPO
from .ppo import TrainConfig as _BaseTrainConfig


@dataclass
class TrainConfig(_BaseTrainConfig):
    recurrent: bool = True
    tbptt_steps: int = 16
    state_carry: float = 1.0


class PPO(_BasePPO):
    """PPO that preserves connectome state and trains it with truncated BPTT."""

    def __init__(self, cfg: TrainConfig):
        if cfg.tbptt_steps < 1:
            raise ValueError("tbptt_steps must be >= 1")
        if not 0.0 <= cfg.state_carry <= 1.0:
            raise ValueError("state_carry must be in [0, 1]")
        super().__init__(cfg)
        if cfg.monitor:
            from ..monitor.publish_v2 import publish_static
            publish_static(self.circuit, self.env, self.policy, cfg)

    def _build_policy(self):
        policy = super()._build_policy()
        if hasattr(policy, "cfg") and hasattr(policy.cfg, "carry"):
            policy.cfg.carry = self.cfg.state_carry if self.cfg.recurrent else 0.0
        return policy

    @torch.no_grad()
    def rollout(self, obs: dict, state: torch.Tensor):
        if not self.cfg.recurrent:
            return super().rollout(obs, state)

        cfg = self.cfg
        T, E, K = cfg.rollout, cfg.n_envs, cfg.tbptt_steps
        buf = {
            "proprio": torch.zeros(T, E, self.env.PROPRIO_DIM, device=self.device),
            "raw": torch.zeros(T, E, self.env.action_dim, device=self.device),
            "logp": torch.zeros(T, E, device=self.device),
            "val": torch.zeros(T, E, device=self.device),
            "rew": torch.zeros(T, E, device=self.device),
            "done": torch.zeros(T, E, device=self.device),
        }
        has_vision = "on" in obs
        if has_vision:
            C = obs["on"].shape[1]
            buf["on"] = torch.zeros(T, E, C, device=self.device)
            buf["off"] = torch.zeros(T, E, C, device=self.device)

        n_chunks = math.ceil(T / K)
        state_starts = torch.empty(
            (n_chunks, *state.shape), dtype=state.dtype, device="cpu"
        )

        for t in range(T):
            if t % K == 0:
                state_starts[t // K].copy_(state.detach().cpu())

            to = self._to_torch(obs)
            buf["proprio"][t] = to["proprio"]
            if has_vision:
                buf["on"][t] = to["on"]
                buf["off"][t] = to["off"]

            action, logp, value, state, raw = self.policy.act(to, state)
            obs, rew, term, trunc, info = self.env.step(action.cpu().numpy())
            done = term | trunc

            buf["raw"][t] = raw
            buf["logp"][t] = logp
            buf["val"][t] = value
            buf["rew"][t] = torch.from_numpy(rew).float().to(self.device)
            buf["done"][t] = torch.from_numpy(done.astype(np.float32)).to(self.device)

            if done.any():
                state = state.clone()
                mask = torch.from_numpy(done).to(state.device)
                state[:, mask] = 0.0
                self._log_episodes(info, done)

            self.global_step += E
            if self.cfg.monitor and (t % 8 == 0):
                self._publish_live(info, state)

        _, _, last_val, _ = self.policy.forward(self._to_torch(obs), state)
        return buf, obs, state, state_starts, last_val

    def update(self, buf, adv, ret, state_starts):
        if not self.cfg.recurrent:
            return super().update(buf, adv, ret, state_starts)

        cfg = self.cfg
        T, E, K = cfg.rollout, cfg.n_envs, cfg.tbptt_steps
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        target_samples = max(1, (T * E) // max(cfg.minibatches, 1))
        env_batch = max(1, min(E, math.ceil(target_samples / K)))

        stats = {"pg": [], "vf": [], "ent": [], "kl": [], "clipfrac": []}
        n_chunks = math.ceil(T / K)

        for _ in range(cfg.epochs):
            chunk_order = np.random.permutation(n_chunks)
            for ci in chunk_order:
                t0, t1 = ci * K, min((ci + 1) * K, T)
                env_order = np.random.permutation(E)

                for es in range(0, E, env_batch):
                    env_np = env_order[es : es + env_batch]
                    env_cpu = torch.from_numpy(env_np.astype(np.int64))
                    env_idx = env_cpu.to(self.device)
                    state = state_starts[ci].index_select(1, env_cpu).to(self.device).detach()

                    logps, ents, values = [], [], []
                    old_logps, advs, rets = [], [], []

                    for t in range(t0, t1):
                        obs = {"proprio": buf["proprio"][t].index_select(0, env_idx)}
                        if "on" in buf:
                            obs["on"] = buf["on"][t].index_select(0, env_idx)
                            obs["off"] = buf["off"][t].index_select(0, env_idx)

                        mean, log_std, value, state = self.policy.forward(obs, state)
                        raw = buf["raw"][t].index_select(0, env_idx)
                        logp, ent = self._logp(mean, log_std, raw)

                        logps.append(logp)
                        ents.append(ent)
                        values.append(value)
                        old_logps.append(buf["logp"][t].index_select(0, env_idx))
                        advs.append(adv[t].index_select(0, env_idx))
                        rets.append(ret[t].index_select(0, env_idx))

                        alive = 1.0 - buf["done"][t].index_select(0, env_idx)
                        state = state * alive.unsqueeze(0)

                    logp = torch.stack(logps)
                    ent = torch.stack(ents)
                    value = torch.stack(values)
                    old_logp = torch.stack(old_logps)
                    a = torch.stack(advs)
                    target = torch.stack(rets)

                    ratio = (logp - old_logp).exp()
                    pg = -torch.min(
                        ratio * a,
                        torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a,
                    ).mean()
                    vf = 0.5 * (value - target).pow(2).mean()
                    ent_m = ent.mean()
                    loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent_m

                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                    self.opt.step()

                    with torch.no_grad():
                        dlog = logp - old_logp
                        kl = ((ratio - 1) - dlog).mean()
                        clipfrac = ((ratio - 1).abs() > cfg.clip).float().mean()
                    stats["pg"].append(pg.item())
                    stats["vf"].append(vf.item())
                    stats["ent"].append(ent_m.item())
                    stats["kl"].append(kl.item())
                    stats["clipfrac"].append(clipfrac.item())

                if stats["kl"] and np.mean(stats["kl"][-max(1, math.ceil(E / env_batch)):]) > cfg.target_kl:
                    break
            else:
                continue
            break

        return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}

    def _publish_live(self, info, state):
        from ..monitor.publish_v2 import publish_frame
        publish_frame(self.env, self.policy, self.circuit, state, info)


def train(cfg: TrainConfig | None = None):
    ppo = PPO(cfg or TrainConfig())
    ppo.train()
    return ppo
