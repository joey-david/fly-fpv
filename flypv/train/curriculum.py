"""Mastery-gated curriculum for long unattended training runs.

Difficulty only moves upward.  A level is promoted once the policy is reliably
clearing it, crashes are under control, and mean flight speed has stopped
improving materially over a short update window.  Promotion regenerates the
course pool at the new geometry and starts a fresh mastery window, so easy-level
statistics cannot leak into the next decision.
"""
from __future__ import annotations

import numpy as np

from .config import TrainConfig
from .ppo import PPO as _BasePPO


class PPO(_BasePPO):
    """PPO with a monotonic, mastery-gated course curriculum."""

    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        self._level_gates: list[float] = []
        self._level_crashes: list[float] = []
        self._level_speeds: list[float] = []

    def _log_episodes(self, info, done):
        super()._log_episodes(info, done)
        ids = np.flatnonzero(done)
        self._level_gates.extend(info["gates_done"][ids].astype(float).tolist())
        self._level_crashes.extend(info["crashed"][ids].astype(float).tolist())
        self._level_gates = self._level_gates[-200:]
        self._level_crashes = self._level_crashes[-200:]

    def _curriculum(self):
        cfg = self.cfg
        if not cfg.curriculum or self.env.cfg.difficulty >= 1.0:
            return

        # One speed observation per rollout/update.  This is deliberately a
        # physical m/s quantity, independent of reward scaling.
        speed = float(np.linalg.norm(self.env.vel, axis=-1).mean())
        self._level_speeds.append(speed)
        self._level_speeds = self._level_speeds[-cfg.curriculum_speed_window :]

        # Require enough completed episodes at *this* level before calling it
        # mastered.  The per-level lists are cleared after every promotion.
        required_episodes = max(50, 3 * cfg.n_envs)
        if len(self._level_gates) < required_episodes:
            return

        gates = np.asarray(self._level_gates[-100:], dtype=float)
        crashes = np.asarray(self._level_crashes[-100:], dtype=float)
        gate_frac = float(gates.mean()) / cfg.env.n_gates
        crash_rate = float(crashes.mean())
        if gate_frac < cfg.curriculum_target or crash_rate > cfg.curriculum_crash_target:
            return

        # Mastery alone is not enough: let the policy exploit a level until its
        # speed has plateaued, then make the geometry harder.  Compare the first
        # and second halves of the recent update window using a relative gain.
        if len(self._level_speeds) < cfg.curriculum_speed_window:
            return
        half = len(self._level_speeds) // 2
        older = float(np.mean(self._level_speeds[:half]))
        newer = float(np.mean(self._level_speeds[half:]))
        speed_gain = (newer - older) / max(abs(older), 0.05)
        if speed_gain > cfg.curriculum_speed_tol:
            return

        old = float(self.env.cfg.difficulty)
        new = min(1.0, old + cfg.curriculum_rate)
        if new <= old:
            return
        self.env.cfg.difficulty = new
        self.env._pool = [self.env._new_course() for _ in range(self.env.cfg.course_pool)]
        self._level_gates.clear()
        self._level_crashes.clear()
        self._level_speeds.clear()
        print(
            f"[curriculum] {old:.2f} -> {new:.2f} | "
            f"gates {gate_frac:.1%} | crash {crash_rate:.1%} | "
            f"speed plateau {speed_gain:+.1%}"
        )


def train(cfg: TrainConfig | None = None):
    trainer = PPO(cfg or TrainConfig())
    return trainer.train()
