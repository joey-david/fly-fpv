"""PPO over the connectome policy.

Two design notes that are not standard boilerplate:

**Why the default policy is stateless across control steps.** The neural state
`h` has one entry per neuron — 28,361 of them — so storing it for every step of
every environment in a rollout would cost hundreds of megabytes and make
minibatching miserable. With `carry = 0` the network instead settles for
`n_iters` synaptic hops from a zeroed state on each control step, which is the
same formulation the connectome-graph-policy literature uses, and it is
consistent with how the circuit was extracted: the subgraph contains exactly the
neurons within a few synapses of a sensor and a few of a muscle. Set
`recurrent=True` for genuine across-step memory; the trainer then switches to
sequence minibatches over environments and replays the state.

**Why the action is squashed after sampling.** The Gaussian is on the raw
pre-tanh action and the log-prob is corrected for the tanh Jacobian, so the
15 wing controls stay inside their physiological travel without the
distribution silently lying about its density at the boundary.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..connectome import build_flight_circuit
from ..env import EnvConfig, HoopRaceEnv
from ..monitor.bus import BUS
from ..policy import ConnectomePolicy, MLPPolicy, PolicyConfig, shuffle_connectome


@dataclass
class TrainConfig:
    total_steps: int = 5_000_000
    rollout: int = 128
    n_envs: int = 32
    epochs: int = 4
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.999   # 200 Hz control, so this is a ~5 s horizon
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.004
    max_grad_norm: float = 0.75
    target_kl: float = 0.03
    recurrent: bool = False
    warmstart_steps: int = 400   # expert transitions per env for behaviour cloning; 0 disables

    arch: str = "connectome"        # connectome | shuffled | erdos | mlp
    scale: str = "flight"           # full | flight | core
    max_neurons: int | None = None
    n_iters: int = 4
    learn_synapses: bool = False

    # curriculum: raise course difficulty once the fly is reliably clearing gates
    curriculum: bool = True
    curriculum_target: float = 0.72   # fraction of gates cleared per episode
    curriculum_rate: float = 0.02

    device: str = "auto"
    seed: int = 0
    run_name: str = "flypv"
    out_dir: str = "runs"
    monitor: bool = True
    monitor_every: int = 1
    save_every: int = 40
    env: EnvConfig = field(default_factory=EnvConfig)


def pick_device(pref: str = "auto") -> str:
    """Prefer the GPU. On Apple Silicon that means Metal via MPS.

    Measured on this model — 28,361 neurons, 534,681 edges, batch 512, forward
    and backward — MPS runs about 1.9x faster than CPU and agrees with it to
    1.2e-07, i.e. float32 exactly. `torch.sparse.mm`, which dominates the cost,
    is supported on MPS and is roughly 7x faster there than on CPU; the manual
    gather-and-index_add_ formulation people reach for when they assume sparse
    is unsupported is *slower* on both devices, so do not "optimise" into it.
    """
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PPO:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = pick_device(cfg.device)

        ecfg = cfg.env
        ecfg.n_envs = cfg.n_envs
        ecfg.seed = cfg.seed

        print(f"[flypv] building the flight circuit (scale={cfg.scale}) ...")
        self.circuit = build_flight_circuit(scale=cfg.scale, max_neurons=cfg.max_neurons)

        self.env = HoopRaceEnv(ecfg, hex_coords=self.circuit.hex_coords)
        self.policy = self._build_policy()
        print(f"[flypv] {cfg.arch}: {self.policy.n_trainable():,} trainable parameters")

        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=1e-5)
        self.out = Path(cfg.out_dir) / cfg.run_name
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(_jsonable(asdict(cfg)), indent=2))

        self.global_step = 0
        self.updates = 0
        self.t_start = time.time()
        self._ep_stats = {"return": [], "gates": [], "crash": [], "length": [], "power": []}

        if cfg.monitor:
            from ..monitor.publish import publish_static
            publish_static(self.circuit, self.env, self.policy, cfg)

    # ------------------------------------------------------------------
    def _build_policy(self):
        cfg = self.cfg
        pc = PolicyConfig(n_iters=cfg.n_iters, carry=0.0 if not cfg.recurrent else 0.5,
                          learn_synapses=cfg.learn_synapses)
        if cfg.arch == "connectome":
            return ConnectomePolicy(self.circuit, self.env.PROPRIO_DIM,
                                    self.env.action_dim, pc, self.device)
        if cfg.arch in ("shuffled", "erdos"):
            mode = "degree" if cfg.arch == "shuffled" else "erdos"
            c = shuffle_connectome(self.circuit, mode, seed=cfg.seed)
            return ConnectomePolicy(c, self.env.PROPRIO_DIM, self.env.action_dim, pc, self.device)
        if cfg.arch == "mlp":
            return MLPPolicy(self.env.PROPRIO_DIM, self.env.n_columns,
                             self.env.action_dim, log_std_init=pc.log_std_init,
                             device=self.device)
        raise ValueError(cfg.arch)

    # ------------------------------------------------------------------
    def _to_torch(self, obs: dict) -> dict:
        return {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}

    @torch.no_grad()
    def rollout(self, obs: dict, state: torch.Tensor):
        cfg = self.cfg
        T, E = cfg.rollout, cfg.n_envs
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

        state0 = state.clone()
        for t in range(T):
            to = self._to_torch(obs)
            buf["proprio"][t] = to["proprio"]
            if has_vision:
                buf["on"][t] = to["on"]; buf["off"][t] = to["off"]

            action, logp, value, state, raw = self.policy.act(to, state)
            a = action.cpu().numpy()
            obs, rew, term, trunc, info = self.env.step(a)
            done = term | trunc

            buf["raw"][t] = raw
            buf["logp"][t] = logp
            buf["val"][t] = value
            buf["rew"][t] = torch.from_numpy(rew).float().to(self.device)
            buf["done"][t] = torch.from_numpy(done.astype(np.float32)).to(self.device)

            if done.any():
                state = state.clone()
                state[:, torch.from_numpy(done)] = 0.0
                self._log_episodes(info, done)
            self.global_step += E

            if self.cfg.monitor and (t % 8 == 0):
                self._publish_live(info, state)

        with torch.no_grad():
            _, _, last_val, _ = self.policy.forward(self._to_torch(obs), state)
        return buf, obs, state, state0, last_val

    def _log_episodes(self, info, done):
        d = np.flatnonzero(done)
        self._ep_stats["return"] += info["episode_return"][d].tolist()
        self._ep_stats["length"] += info["episode_length"][d].tolist()
        self._ep_stats["gates"] += info["gates_done"][d].tolist()
        self._ep_stats["crash"] += info["crashed"][d].astype(float).tolist()
        for k in self._ep_stats:
            if len(self._ep_stats[k]) > 400:
                self._ep_stats[k] = self._ep_stats[k][-400:]

    # ------------------------------------------------------------------
    def gae(self, buf, last_val):
        cfg = self.cfg
        T = cfg.rollout
        adv = torch.zeros_like(buf["rew"])
        gae = torch.zeros(cfg.n_envs, device=self.device)
        for t in reversed(range(T)):
            nextval = last_val if t == T - 1 else buf["val"][t + 1]
            nonterm = 1.0 - buf["done"][t]
            delta = buf["rew"][t] + cfg.gamma * nextval * nonterm - buf["val"][t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
            adv[t] = gae
        return adv, adv + buf["val"]

    def _logp(self, mean, log_std, raw):
        from ..policy.connectome_policy import squashed_log_prob
        d = torch.distributions.Normal(mean, log_std.exp())
        return squashed_log_prob(d, raw), d.entropy().sum(-1)

    def update(self, buf, adv, ret, state0):
        cfg = self.cfg
        T, E = cfg.rollout, cfg.n_envs
        flat = lambda x: x.reshape(T * E, *x.shape[2:])
        b = {k: flat(v) for k, v in buf.items()}
        b_adv, b_ret = flat(adv), flat(ret)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = T * E
        mb = n // cfg.minibatches
        idx = np.arange(n)
        stats = {"pg": [], "vf": [], "ent": [], "kl": [], "clipfrac": []}

        for ep in range(cfg.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, mb):
                j = torch.from_numpy(idx[s : s + mb]).to(self.device)
                obs = {"proprio": b["proprio"][j]}
                if "on" in b:
                    obs["on"] = b["on"][j]; obs["off"] = b["off"][j]

                st = self.policy.init_state(len(j))
                mean, log_std, value, _ = self.policy.forward(obs, st)
                logp, ent = self._logp(mean, log_std, b["raw"][j])

                ratio = (logp - b["logp"][j]).exp()
                a = b_adv[j]
                pg = -torch.min(ratio * a,
                                torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vf = 0.5 * (value - b_ret[j]).pow(2).mean()
                ent_m = ent.mean()
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent_m

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - b["logp"][j])).mean()
                stats["pg"].append(pg.item()); stats["vf"].append(vf.item())
                stats["ent"].append(ent_m.item()); stats["kl"].append(kl.item())
                stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())

            if np.mean(stats["kl"][-cfg.minibatches:]) > cfg.target_kl:
                break
        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ------------------------------------------------------------------
    def _publish_live(self, info, state):
        from ..monitor.publish import publish_frame
        publish_frame(self.env, self.policy, self.circuit, state, info)

    def _curriculum(self):
        cfg = self.cfg
        if not cfg.curriculum or not self._ep_stats["gates"]:
            return
        frac = float(np.mean(self._ep_stats["gates"][-100:])) / cfg.env.n_gates
        step = cfg.curriculum_rate * (1 if frac > cfg.curriculum_target else -0.5)
        self.env.cfg.difficulty = float(np.clip(self.env.cfg.difficulty + step, 0.05, 1.0))
        # regenerate the course pool so the new difficulty actually takes effect
        if self.updates % 10 == 0:
            self.env._pool = [self.env._new_course() for _ in range(self.env.cfg.course_pool)]

    def train(self):
        cfg = self.cfg
        if cfg.warmstart_steps:
            from .imitate import warm_start
            warm_start(self.policy, self.env, steps=cfg.warmstart_steps)
            # the cloning optimiser is discarded; PPO starts with fresh moments
            self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=1e-5)
        obs = self.env.reset()
        state = self.policy.init_state(cfg.n_envs)

        while self.global_step < cfg.total_steps:
            t0 = time.time()
            buf, obs, state, state0, last_val = self.rollout(obs, state)
            adv, ret = self.gae(buf, last_val)
            stats = self.update(buf, adv, ret, state0)
            self.updates += 1
            self._curriculum()

            dt = time.time() - t0
            fps = cfg.rollout * cfg.n_envs / dt
            ep = {k: (float(np.mean(v[-100:])) if v else 0.0) for k, v in self._ep_stats.items()}
            BUS.record(
                step=self.global_step, update=self.updates, fps=fps,
                ep_return=ep["return"], ep_gates=ep["gates"], ep_length=ep["length"],
                crash_rate=ep["crash"], difficulty=self.env.cfg.difficulty,
                policy_loss=stats["pg"], value_loss=stats["vf"],
                entropy=stats["ent"], kl=stats["kl"], clipfrac=stats["clipfrac"],
                elapsed=time.time() - self.t_start,
            )
            print(f"[{self.global_step:>9,}] ret {ep['return']:7.2f} | gates {ep['gates']:4.2f}"
                  f"/{cfg.env.n_gates} | crash {ep['crash']:4.2f} | diff {self.env.cfg.difficulty:.2f}"
                  f" | ent {stats['ent']:6.2f} | kl {stats['kl']:.4f} | {fps:6.0f} fps")

            if self.updates % cfg.save_every == 0:
                self.save()
        self.save()

    def save(self):
        p = self.out / "policy.pt"
        torch.save({"policy": self.policy.state_dict(), "step": self.global_step,
                    "cfg": _jsonable(asdict(self.cfg))}, p)
        return p


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, (np.integer, np.floating)):
        return d.item()
    return d


def train(cfg: TrainConfig | None = None):
    ppo = PPO(cfg or TrainConfig())
    ppo.train()
    return ppo
