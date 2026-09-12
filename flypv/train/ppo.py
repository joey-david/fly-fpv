"""Plain feed-forward PPO for connectome-constrained flight policies."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..connectome import build_flight_circuit
from ..env import HoopRaceEnv
from ..monitor.bus import BUS
from ..policy import ConnectomePolicy, MLPPolicy, PolicyConfig, shuffle_connectome
from .checkpoint import CheckpointManager
from .config import TrainConfig, save_yaml


STRUCTURAL_RESUME_FIELDS = ("arch", "scale", "max_neurons", "learn_synapses")


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PPO:
    def __init__(self, cfg: TrainConfig):
        cfg.validate()
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        cfg.env.n_envs = cfg.n_envs
        cfg.env.seed = cfg.seed
        print(f"[flypv] building the flight circuit (scale={cfg.scale}) ...")
        self.circuit = build_flight_circuit(scale=cfg.scale, max_neurons=cfg.max_neurons)
        self.env = HoopRaceEnv(cfg.env, hex_coords=self.circuit.hex_coords)
        self.policy = self._build_policy()
        print(f"[flypv] {cfg.arch}: {self.policy.n_trainable():,} trainable parameters")

        self.opt = self._new_optimizer()
        self.out = Path(cfg.out_dir) / cfg.run_name
        self.out.mkdir(parents=True, exist_ok=True)
        self.checkpoints = CheckpointManager(
            self.out, keep=cfg.keep_checkpoints, best_metric=cfg.best_metric
        )

        self.global_step = 0
        self.updates = 0
        self.t_start = time.time()
        self.elapsed_before_resume = 0.0
        self._ep_stats = {"return": [], "gates": [], "crash": [], "length": []}
        self._last_metrics: dict[str, float] = {}
        self._resumed = False

        if cfg.resume:
            path = self.checkpoints.resolve(self.out, cfg.resume)
            self._load_checkpoint(path)

        resolved = TrainConfig.from_dict(cfg.to_dict())
        resolved.resume = None
        resolved.reset_optimizer = False
        save_yaml(resolved, self.out / "config.yaml")

        if cfg.monitor:
            from ..monitor.publish import publish_static
            publish_static(self.circuit, self.env, self.policy, cfg)

    def _new_optimizer(self):
        return torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr, eps=1e-5)

    def _build_policy(self):
        cfg = self.cfg
        pc = PolicyConfig(n_iters=cfg.n_iters, learn_synapses=cfg.learn_synapses)
        if cfg.arch == "connectome":
            return ConnectomePolicy(
                self.circuit, self.env.PROPRIO_DIM, self.env.action_dim, pc, self.device
            )
        if cfg.arch in ("shuffled", "erdos"):
            mode = "degree" if cfg.arch == "shuffled" else "erdos"
            circuit = shuffle_connectome(self.circuit, mode, seed=cfg.seed)
            return ConnectomePolicy(
                circuit, self.env.PROPRIO_DIM, self.env.action_dim, pc, self.device
            )
        if cfg.arch == "mlp":
            return MLPPolicy(
                self.env.PROPRIO_DIM,
                self.env.n_columns,
                self.env.action_dim,
                log_std_init=pc.log_std_init,
                device=self.device,
            )
        raise ValueError(cfg.arch)

    def _load_checkpoint(self, path: Path) -> None:
        payload = CheckpointManager.load(path, map_location="cpu")
        saved_cfg = payload.get("cfg") or {}
        for key in STRUCTURAL_RESUME_FIELDS:
            old = saved_cfg.get(key)
            new = getattr(self.cfg, key)
            if old != new:
                raise ValueError(
                    f"cannot resume with {key}={new!r}; checkpoint was built with {key}={old!r}"
                )
        if self.cfg.arch == "mlp":
            old_vision = (saved_cfg.get("env") or {}).get("vision", True)
            if old_vision != self.cfg.env.vision:
                raise ValueError("MLP checkpoints cannot change vision mode on resume")

        self.policy.load_state_dict(payload["policy"], strict=True)
        if self.cfg.reset_optimizer:
            self.opt = self._new_optimizer()
        else:
            self.opt.load_state_dict(payload["optimizer"])
            for group in self.opt.param_groups:
                group["lr"] = self.cfg.lr

        self.global_step = int(payload.get("step", 0))
        self.updates = int(payload.get("updates", 0))
        self.env.cfg.difficulty = float(payload.get("difficulty", self.env.cfg.difficulty))
        self.elapsed_before_resume = float(payload.get("elapsed", 0.0))
        loaded_stats = payload.get("ep_stats") or {}
        for key in self._ep_stats:
            self._ep_stats[key] = list(loaded_stats.get(key, []))[-400:]
        CheckpointManager.restore_rng(payload)
        self._resumed = True
        print(
            f"[flypv] resumed {path.name} at step {self.global_step:,}, "
            f"update {self.updates:,}, difficulty {self.env.cfg.difficulty:.2f}"
        )

    def _to_torch(self, obs: dict) -> dict:
        return {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}

    def _empty_buffer(self, obs: dict) -> dict[str, torch.Tensor]:
        T, E = self.cfg.rollout, self.cfg.n_envs
        buf = {
            "proprio": torch.zeros(T, E, self.env.PROPRIO_DIM, device=self.device),
            "raw": torch.zeros(T, E, self.env.action_dim, device=self.device),
            "logp": torch.zeros(T, E, device=self.device),
            "val": torch.zeros(T, E, device=self.device),
            "rew": torch.zeros(T, E, device=self.device),
            "done": torch.zeros(T, E, device=self.device),
            "timeout_value": torch.zeros(T, E, device=self.device),
        }
        if "on" in obs:
            C = obs["on"].shape[1]
            buf["on"] = torch.zeros(T, E, C, device=self.device)
            buf["off"] = torch.zeros(T, E, C, device=self.device)
        return buf

    @torch.no_grad()
    def rollout(self, obs: dict):
        cfg = self.cfg
        T, E = cfg.rollout, cfg.n_envs
        buf = self._empty_buffer(obs)
        last_h = self.policy.init_state(E)

        for t in range(T):
            to = self._to_torch(obs)
            buf["proprio"][t] = to["proprio"]
            if "on" in buf:
                buf["on"][t] = to["on"]
                buf["off"][t] = to["off"]

            # Every decision is independent. The connectome is unrolled only
            # inside this forward pass; no neural state is carried across steps.
            state0 = self.policy.init_state(E)
            action, logp, value, last_h, raw = self.policy.act(to, state0)
            obs, rew, term, trunc, info = self.env.step(action.cpu().numpy())
            done = term | trunc

            buf["raw"][t] = raw
            buf["logp"][t] = logp
            buf["val"][t] = value
            buf["rew"][t] = torch.from_numpy(rew).float().to(self.device)
            buf["done"][t] = torch.from_numpy(done.astype(np.float32)).to(self.device)

            trunc_only = trunc & ~term
            if trunc_only.any() and "final_obs" in info:
                ids_np = np.flatnonzero(trunc_only)
                ids = torch.from_numpy(ids_np.astype(np.int64)).to(self.device)
                final_obs = {
                    k: torch.from_numpy(v[ids_np]).to(self.device)
                    for k, v in info["final_obs"].items()
                }
                _, _, final_value, _ = self.policy.forward(
                    final_obs, self.policy.init_state(len(ids_np))
                )
                buf["timeout_value"][t].index_copy_(0, ids, final_value)

            if done.any():
                self._log_episodes(info, done)

            self.global_step += E
            if cfg.monitor and t % 8 == 0:
                self._publish_live(info, last_h)

        _, _, last_val, _ = self.policy.forward(
            self._to_torch(obs), self.policy.init_state(E)
        )
        return buf, obs, last_h, last_val

    def _log_episodes(self, info, done):
        ids = np.flatnonzero(done)
        self._ep_stats["return"] += info["episode_return"][ids].tolist()
        self._ep_stats["length"] += info["episode_length"][ids].tolist()
        self._ep_stats["gates"] += info["gates_done"][ids].tolist()
        self._ep_stats["crash"] += info["crashed"][ids].astype(float).tolist()
        for key in self._ep_stats:
            if len(self._ep_stats[key]) > 400:
                self._ep_stats[key] = self._ep_stats[key][-400:]

    def gae(self, buf, last_val):
        cfg = self.cfg
        adv = torch.zeros_like(buf["rew"])
        gae = torch.zeros(cfg.n_envs, device=self.device)
        for t in reversed(range(cfg.rollout)):
            nextval = last_val if t == cfg.rollout - 1 else buf["val"][t + 1]
            nonterm = 1.0 - buf["done"][t]
            delta = (
                buf["rew"][t]
                + cfg.gamma * nextval * nonterm
                + cfg.gamma * buf["timeout_value"][t]
                - buf["val"][t]
            )
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
            adv[t] = gae
        return adv, adv + buf["val"]

    def _logp(self, mean, log_std, raw):
        from ..policy.connectome_policy import squashed_log_prob
        dist = torch.distributions.Normal(mean, log_std.exp())
        return squashed_log_prob(dist, raw), dist.entropy().sum(-1)

    def update(self, buf, adv, ret):
        cfg = self.cfg
        T, E = cfg.rollout, cfg.n_envs

        def flat(x):
            return x.reshape(T * E, *x.shape[2:])

        b = {k: flat(v) for k, v in buf.items()}
        b_adv, b_ret = flat(adv), flat(ret)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        n = T * E
        mb = max(1, n // cfg.minibatches)
        order = np.arange(n)
        stats = self._empty_update_stats()

        for _ in range(cfg.epochs):
            np.random.shuffle(order)
            epoch_kls = []
            for start in range(0, n, mb):
                idx = torch.from_numpy(order[start:start + mb]).to(self.device)
                batch_obs = {"proprio": b["proprio"][idx]}
                if "on" in b:
                    batch_obs["on"] = b["on"][idx]
                    batch_obs["off"] = b["off"][idx]
                mean, log_std, value, _ = self.policy.forward(
                    batch_obs, self.policy.init_state(len(idx))
                )
                logp, ent = self._logp(mean, log_std, b["raw"][idx])
                batch_stats = self._optimize_batch(
                    logp, ent, value, b["logp"][idx], b_adv[idx], b_ret[idx]
                )
                self._append_stats(stats, batch_stats)
                epoch_kls.append(batch_stats["kl"])
            if epoch_kls and float(np.mean(epoch_kls)) > cfg.target_kl:
                break
        return self._mean_stats(stats)

    def _optimize_batch(self, logp, ent, value, old_logp, adv, target):
        cfg = self.cfg
        ratio = (logp - old_logp).exp()
        pg = -torch.min(
            ratio * adv,
            torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv,
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
        return {"pg": pg.item(), "vf": vf.item(), "ent": ent_m.item(),
                "kl": kl.item(), "clipfrac": clipfrac.item()}

    @staticmethod
    def _empty_update_stats():
        return {"pg": [], "vf": [], "ent": [], "kl": [], "clipfrac": []}

    @staticmethod
    def _append_stats(stats, values):
        for key, value in values.items():
            stats[key].append(value)

    @staticmethod
    def _mean_stats(stats):
        return {key: float(np.mean(values)) if values else 0.0 for key, values in stats.items()}

    def _publish_live(self, info, state):
        from ..monitor.publish import publish_frame
        publish_frame(self.env, self.policy, self.circuit, state, info)

    def _curriculum(self):
        cfg = self.cfg
        if not cfg.curriculum or not self._ep_stats["gates"]:
            return
        frac = float(np.mean(self._ep_stats["gates"][-100:])) / cfg.env.n_gates
        step = cfg.curriculum_rate * (1.0 if frac > cfg.curriculum_target else -0.5)
        self.env.cfg.difficulty = float(np.clip(self.env.cfg.difficulty + step, 0.05, 1.0))
        if self.updates % 10 == 0:
            self.env._pool = [self.env._new_course() for _ in range(self.env.cfg.course_pool)]

    def _summary_metrics(self, stats: dict[str, float], fps: float) -> dict[str, float]:
        ep = {key: float(np.mean(values[-100:])) if values else 0.0
              for key, values in self._ep_stats.items()}
        return {
            "step": float(self.global_step), "update": float(self.updates), "fps": float(fps),
            "ep_return": ep["return"], "ep_gates": ep["gates"], "ep_length": ep["length"],
            "crash_rate": ep["crash"], "difficulty": float(self.env.cfg.difficulty),
            "policy_loss": stats["pg"], "value_loss": stats["vf"], "entropy": stats["ent"],
            "kl": stats["kl"], "clipfrac": stats["clipfrac"],
            "elapsed": self.elapsed_before_resume + time.time() - self.t_start,
        }

    def train(self):
        cfg = self.cfg
        if cfg.warmstart_steps and not self._resumed:
            from .imitate import warm_start
            warm_start(self.policy, self.env, steps=cfg.warmstart_steps)
            self.opt = self._new_optimizer()

        obs = self.env.reset()
        try:
            while self.global_step < cfg.total_steps:
                t0 = time.time()
                buf, obs, _, last_val = self.rollout(obs)
                adv, ret = self.gae(buf, last_val)
                stats = self.update(buf, adv, ret)
                self.updates += 1
                self._curriculum()

                fps = cfg.rollout * cfg.n_envs / max(time.time() - t0, 1e-9)
                metrics = self._summary_metrics(stats, fps)
                self._last_metrics = metrics
                BUS.record(**metrics)
                print(
                    f"[{self.global_step:>9,}] ret {metrics['ep_return']:7.2f} | "
                    f"gates {metrics['ep_gates']:4.2f}/{cfg.env.n_gates} | "
                    f"crash {metrics['crash_rate']:4.2f} | diff {metrics['difficulty']:.2f} | "
                    f"ent {stats['ent']:6.2f} | kl {stats['kl']:.4f} | {fps:6.0f} fps"
                )
                if cfg.save_every and self.updates % cfg.save_every == 0:
                    self.save("periodic")
        except KeyboardInterrupt:
            path = self.save("interrupt")
            print(f"\n[flypv] interrupted; checkpoint saved -> {path}")
            return self

        path = self.save("final")
        print(f"[flypv] training complete; checkpoint saved -> {path}")
        return self

    def save(self, reason: str = "manual") -> Path:
        cfg_snapshot = self.cfg.to_dict()
        cfg_snapshot["resume"] = None
        cfg_snapshot["reset_optimizer"] = False
        elapsed = self.elapsed_before_resume + time.time() - self.t_start
        return self.checkpoints.save(
            policy=self.policy, optimizer=self.opt, cfg=cfg_snapshot,
            global_step=self.global_step, updates=self.updates,
            difficulty=self.env.cfg.difficulty, ep_stats=self._ep_stats,
            metrics=self._last_metrics, elapsed=elapsed, reason=reason,
        )


def train(cfg: TrainConfig | None = None):
    trainer = PPO(cfg or TrainConfig())
    return trainer.train()
