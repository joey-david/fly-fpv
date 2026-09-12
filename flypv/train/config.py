"""Training configuration and YAML serialization."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..env import EnvConfig


@dataclass
class TrainConfig:
    # rollout / PPO
    total_steps: int = 5_000_000
    rollout: int = 128
    n_envs: int = 32
    epochs: int = 4
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.004
    max_grad_norm: float = 0.75
    target_kl: float = 0.03

    # connectome dynamics
    recurrent: bool = True
    tbptt_steps: int = 16
    state_carry: float = 1.0
    n_iters: int = 4
    learn_synapses: bool = False
    warmstart_steps: int = 400
    input_norm_power: float = 0.5

    # architecture / task
    arch: str = "connectome"  # connectome | shuffled | erdos | mlp
    scale: str = "flight"  # full | flight | core
    circuit_hops: int = 3
    recurrent_closure_hops: int = 1
    max_neurons: int | None = None
    curriculum: bool = True
    curriculum_target: float = 0.72
    curriculum_rate: float = 0.02
    curriculum_crash_target: float = 0.30
    curriculum_speed_window: int = 12
    curriculum_speed_tol: float = 0.02

    # run management
    device: str = "auto"
    seed: int = 0
    run_name: str = "flypv"
    out_dir: str = "runs"
    monitor: bool = True
    save_every: int = 20
    keep_checkpoints: int = 5
    best_metric: str = "ep_gates"
    resume: str | None = None
    reset_optimizer: bool = False

    env: EnvConfig = field(default_factory=EnvConfig)

    def validate(self) -> None:
        if self.total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if self.rollout < 1 or self.n_envs < 1:
            raise ValueError("rollout and n_envs must be >= 1")
        if self.epochs < 1 or self.minibatches < 1:
            raise ValueError("epochs and minibatches must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 0.0 < self.clip < 1.0:
            raise ValueError("clip must be in (0, 1)")
        if self.tbptt_steps < 1:
            raise ValueError("tbptt_steps must be >= 1")
        if not 0.0 <= self.state_carry <= 1.0:
            raise ValueError("state_carry must be in [0, 1]")
        if self.n_iters < 1:
            raise ValueError("n_iters must be >= 1")
        if not 0.0 <= self.input_norm_power <= 1.0:
            raise ValueError("input_norm_power must be in [0, 1]")
        if self.circuit_hops < 1:
            raise ValueError("circuit_hops must be >= 1")
        if self.recurrent_closure_hops < 0:
            raise ValueError("recurrent_closure_hops must be >= 0")
        if self.max_neurons is not None and self.max_neurons < 1:
            raise ValueError("max_neurons must be >= 1")
        if not 0.0 <= self.curriculum_target <= 1.0:
            raise ValueError("curriculum_target must be in [0, 1]")
        if self.curriculum_rate <= 0:
            raise ValueError("curriculum_rate must be > 0")
        if not 0.0 <= self.curriculum_crash_target <= 1.0:
            raise ValueError("curriculum_crash_target must be in [0, 1]")
        if self.curriculum_speed_window < 4:
            raise ValueError("curriculum_speed_window must be >= 4")
        if self.curriculum_speed_tol < 0:
            raise ValueError("curriculum_speed_tol must be >= 0")
        if self.save_every < 0 or self.keep_checkpoints < 1:
            raise ValueError("save_every must be >= 0 and keep_checkpoints >= 1")
        if self.arch not in {"connectome", "shuffled", "erdos", "mlp"}:
            raise ValueError(f"unknown architecture: {self.arch}")
        if self.scale not in {"full", "flight", "core"}:
            raise ValueError(f"unknown circuit scale: {self.scale}")
        if self.best_metric not in {"ep_gates", "ep_return"}:
            raise ValueError("best_metric must be 'ep_gates' or 'ep_return'")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        if not isinstance(data, dict):
            raise TypeError("training config must be a mapping")
        data = dict(data)
        data.pop("monitor_every", None)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown training config keys: {', '.join(sorted(unknown))}")
        values = dict(data)
        env = values.pop("env", None)
        if env is not None:
            env = dict(env)
        cfg = cls(**values)
        if env is not None:
            env_known = {f.name for f in fields(EnvConfig)}
            env_unknown = set(env) - env_known
            if env_unknown:
                raise ValueError(f"unknown env config keys: {', '.join(sorted(env_unknown))}")
            gate = env.pop("gate_spec", None)
            eye = env.pop("eye", None)
            ecfg = EnvConfig(**env)
            if gate is not None:
                for k, v in gate.items():
                    setattr(ecfg.gate_spec, k, tuple(v) if isinstance(v, list) else v)
            if eye is not None:
                for k, v in eye.items():
                    setattr(ecfg.eye, k, tuple(v) if isinstance(v, list) else v)
            cfg.env = ecfg
        cfg.validate()
        return cfg


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    return {} if data is None else data


def save_yaml(cfg: TrainConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))


def merge_config(base: TrainConfig, overrides: dict[str, Any]) -> TrainConfig:
    """Return a validated config with a flat set of top-level/env overrides."""
    data = base.to_dict()
    overrides = dict(overrides)
    env_overrides = overrides.pop("env", None)
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    if env_overrides:
        data.setdefault("env", {}).update({k: v for k, v in env_overrides.items() if v is not None})
    return TrainConfig.from_dict(data)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
