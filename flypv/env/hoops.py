"""The task: fly a 3D hoop course, fast, cheaply, and without thrashing.

A vectorised 6-DOF flapping-flight environment. The plant is the blade-element
model in `flypv.wing`, driven by the wing pattern generator; the agent's action
is the 15-dimensional vector of things a real fly's flight muscles can change.

The three words in the brief map onto three reward terms:

  *fast*      — progress along the course and speed through each hoop
  *efficient* — a penalty on aerodynamic power, in units of hovering power, so
                the agent is charged for the energy it actually spends
  *elegant*   — penalties on control jerk and on body angular rate, which is
                what separates a clean racing line from a wobble that happens
                to get through the gate

Nothing here needs MuJoCo. See `flypv/env/flybody_adapter.py` for swapping in
the DeepMind/Janelia musculoskeletal model if you want contact and full body
dynamics instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..wing import (DROSOPHILA, FlyMorphology, N_WING_CONTROLS,
                    WingPatternGenerator, blade_element_forces)
from ..wing.wpg import ABDOMEN_COM_ARM
from .course import Course, GateSpec, make_course
from .vision import CompoundEye, EyeConfig


@dataclass
class EnvConfig:
    n_envs: int = 16
    n_gates: int = 8
    control_hz: float = 200.0   # one decision per wingbeat, as a real fly steers
    aero_substeps: int = 12
    episode_seconds: float = 6.0
    difficulty: float = 0.4
    vision_every: int = 2          # render the eye every N control steps (100 Hz)
    visible_gates: int = 4         # only ray-cast the gates that can be in front of you
    course_pool: int = 256         # pre-generated courses to resample on reset
    arena_radius: float = 3.0
    ceiling: float = 1.2
    seed: int = 0
    vision: bool = True
    gate_spec: GateSpec = field(default_factory=GateSpec)
    eye: EyeConfig = field(default_factory=EyeConfig)

    # reward weights
    w_progress: float = 30.0
    w_gate: float = 12.0
    w_center: float = 6.0
    w_speed: float = 1.5
    w_energy: float = 0.035
    w_jerk: float = 0.60
    w_spin: float = 0.06
    w_upright: float = 0.25
    w_alive: float = 0.02
    r_crash: float = -20.0
    r_finish: float = 60.0
    r_miss: float = -4.0


class _GateArray:
    """Per-environment gate geometry, (E, G, ...)."""

    def __init__(self, courses: list[Course]):
        self.center = np.stack([c.center for c in courses])
        self.normal = np.stack([c.normal for c in courses])
        self.r_in = np.stack([c.r_in for c in courses])
        self.r_out = np.stack([c.r_out for c in courses])

    def assign(self, i: int, c: Course):
        self.center[i] = c.center
        self.normal[i] = c.normal
        self.r_in[i] = c.r_in
        self.r_out[i] = c.r_out


class HoopRaceEnv:
    """Vectorised flapping-flight hoop racing.

    Observations are a dict:
      ``proprio`` (E, 40)  — halteres, ocelli/gravity, airspeed, wing phase,
                             previous action, and the next two gates in body frame
      ``on`` / ``off`` (E, n_columns) — the lamina's ON and OFF channels, one
                             value per real ommatidium, ready to be written
                             straight onto L1/L5/Mi1 and L2/L3/Tm1/Tm2/Tm9.
    """

    PROPRIO_DIM = 40

    def __init__(self, cfg: EnvConfig | None = None, hex_coords: dict | None = None,
                 morph: FlyMorphology = DROSOPHILA):
        self.cfg = cfg or EnvConfig()
        self.m = morph
        E = self.cfg.n_envs
        self.rng = np.random.default_rng(self.cfg.seed)
        self.dt = 1.0 / self.cfg.control_hz
        self.max_steps = int(self.cfg.episode_seconds * self.cfg.control_hz)

        self.wpg = WingPatternGenerator(E, morph)
        self.eye = (CompoundEye(hex_coords, E, self.cfg.eye)
                    if (self.cfg.vision and hex_coords) else None)
        self.n_columns = self.eye.n_columns if self.eye else 0

        # Generating a course costs more than a physics step, and episodes end
        # often early in training, so draw from a pre-generated pool instead.
        self._pool = [self._new_course() for _ in range(self.cfg.course_pool)]
        self.courses = [self._pool[i % len(self._pool)] for i in range(E)]
        self.gates = _GateArray(self.courses)

        self.pos = np.zeros((E, 3))
        self.vel = np.zeros((E, 3))
        self.quat = np.zeros((E, 4)); self.quat[:, 0] = 1.0
        self.omega = np.zeros((E, 3))
        # Halteres are mechanical filters: they report the beat-averaged body
        # rate, not the several-thousand-deg/s wobble that happens *within* a
        # wingbeat. Everything that reads angular velocity reads this instead.
        self.omega_slow = np.zeros((E, 3))
        self.next_gate = np.zeros(E, dtype=np.int64)
        self.step_count = np.zeros(E, dtype=np.int64)
        self.prev_action = np.zeros((E, N_WING_CONTROLS))
        self.prev_dist = np.zeros(E)
        self.ep_return = np.zeros(E)
        self.ep_power = np.zeros(E)
        self.last_diag: dict = {}
        self.last_lum = np.zeros((E, max(self.n_columns, 1)), dtype=np.float32)
        self.last_on = np.zeros((E, max(self.n_columns, 1)), dtype=np.float32)
        self.last_off = np.zeros((E, max(self.n_columns, 1)), dtype=np.float32)
        self._vis_tick = 0
        # A wingbeat is 208 Hz and the monitor samples at ~25 Hz, so anything
        # read once per control step aliases into a flat line. Keep a short
        # ring buffer written at *substep* resolution instead — about two full
        # strokes — so the dashboard shows the real waveform.
        self._scope_n = 96
        self._scope = np.zeros((self._scope_n, 5), dtype=np.float32)
        self._scope_i = 0

        # hovering aerodynamic power, the unit the energy penalty is charged in
        self.p_hover = self._measure_hover_power()

        self.reset()

    # ------------------------------------------------------------------ setup
    def _new_course(self) -> Course:
        return make_course(self.cfg.n_gates, self.cfg.gate_spec,
                           seed=int(self.rng.integers(1 << 31)),
                           difficulty=self.cfg.difficulty)

    def _measure_hover_power(self) -> float:
        w = WingPatternGenerator(1, self.m)
        w.set_controls(np.zeros((1, N_WING_CONTROLS)))
        ps = []
        for _ in range(256):
            ws = w.advance(1.0 / (self.m.f_nominal * 256))
            _, _, d = blade_element_forces(
                self.m, ws.phi, ws.theta, ws.alpha, ws.dphi, ws.dtheta, ws.dalpha,
                np.zeros((1, 3)), np.zeros((1, 3)), ws.stroke_tilt)
            ps.append(d["power"][0])
        return float(np.mean(ps))

    # ------------------------------------------------------------------ reset
    def reset(self, mask: np.ndarray | None = None):
        E = self.cfg.n_envs
        mask = np.ones(E, dtype=bool) if mask is None else mask
        idx = np.flatnonzero(mask)
        for i in idx:
            self.courses[i] = self._pool[int(self.rng.integers(len(self._pool)))]
            self.gates.assign(i, self.courses[i])
            self.pos[i] = self.courses[i].start_pos
            d = self.courses[i].start_dir
            self.quat[i] = _quat_from_forward(d)
            # launched at a realistic cruising speed, slightly perturbed
            self.vel[i] = d * self.rng.uniform(0.15, 0.35) + self.rng.normal(0, 0.02, 3)
        self.omega[idx] = self.rng.normal(0, 0.5, (len(idx), 3))
        self.omega_slow[idx] = self.omega[idx]
        self.next_gate[idx] = 0
        self.step_count[idx] = 0
        self.prev_action[idx] = 0.0
        self.ep_return[idx] = 0.0
        self.ep_power[idx] = 0.0
        self.wpg.reset(mask, self.rng)
        self.prev_dist[idx] = np.linalg.norm(
            self.gates.center[idx, 0] - self.pos[idx], axis=-1)
        return self._observe(reset_mask=mask)

    # ------------------------------------------------------------------- step
    def step(self, action: np.ndarray):
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.wpg.set_controls(action)

        sub_dt = self.dt / cfg.aero_substeps
        power_acc = np.zeros(cfg.n_envs)
        pos0 = self.pos.copy()

        for _ in range(cfg.aero_substeps):
            ws = self.wpg.advance(sub_dt)
            R_bw = _quat_to_mat(self.quat)                 # body -> world
            v_body = np.einsum("eji,ej->ei", R_bw, self.vel)

            com = np.zeros((cfg.n_envs, 3))
            com[:, 0] = -ABDOMEN_COM_ARM * ws.abdomen
            F_b, T_b, diag = blade_element_forces(
                self.m, ws.phi, ws.theta, ws.alpha, ws.dphi, ws.dtheta, ws.dalpha,
                v_body, self.omega, ws.stroke_tilt, com_shift=com)
            power_acc += diag["power"]

            F_w = np.einsum("eij,ej->ei", R_bw, F_b)
            F_w[:, 2] -= self.m.weight
            self.vel += (F_w / self.m.mass) * sub_dt
            self.pos += self.vel * sub_dt

            I = self.m.inertia
            gyro = np.cross(self.omega, self.omega * I)
            self.omega += ((T_b - gyro) / I) * sub_dt
            self.quat = _quat_integrate(self.quat, self.omega, sub_dt)

            self._scope[self._scope_i] = (ws.phi[0, 0], ws.phi[0, 1],
                                          ws.alpha[0, 0], ws.alpha[0, 1],
                                          diag["aoa"][0, 0])
            self._scope_i = (self._scope_i + 1) % self._scope_n

        # haltere low-pass, tau = 10 ms (about two wingbeats)
        a = self.dt / (0.010 + self.dt)
        self.omega_slow += a * (self.omega - self.omega_slow)

        self.last_diag = diag
        power = power_acc / cfg.aero_substeps
        self.ep_power += power * self.dt
        self.step_count += 1

        reward, terminated, info = self._reward(action, power, pos0)
        truncated = self.step_count >= self.max_steps
        self.ep_return += reward
        self.prev_action = action

        done = terminated | truncated
        obs = self._observe(reset_mask=None)
        info["episode_return"] = self.ep_return.copy()
        info["episode_length"] = self.step_count.copy()
        if done.any():
            # autoreset, the convention PPO rollout collection expects
            obs_final = {k: v.copy() for k, v in obs.items()}
            info["final_obs"] = obs_final
            obs = self.reset(done)
        return obs, reward, terminated, truncated, info

    # ----------------------------------------------------------------- reward
    def _reward(self, action, power, pos0):
        cfg = self.cfg
        E = cfg.n_envs
        ar = np.arange(E)
        gi = np.minimum(self.next_gate, cfg.n_gates - 1)
        c = self.gates.center[ar, gi]
        n = self.gates.normal[ar, gi]
        r_in = self.gates.r_in[ar, gi]
        r_out = self.gates.r_out[ar, gi]

        R_bw = _quat_to_mat(self.quat)

        # --- gate crossing: sign change of the plane-signed distance ----------
        s0 = np.einsum("ej,ej->e", pos0 - c, n)
        s1 = np.einsum("ej,ej->e", self.pos - c, n)
        crossed = (s0 < 0) & (s1 >= 0)
        # where in the plane did the path cross?
        t = np.where(np.abs(s1 - s0) > 1e-12, -s0 / (s1 - s0 + 1e-12), 0.0)[:, None]
        hit = pos0 + t * (self.pos - pos0)
        rad = np.linalg.norm(hit - c, axis=-1)

        passed = crossed & (rad < r_in)
        clipped = crossed & (rad >= r_in) & (rad <= r_out)     # hit the ring itself
        missed = crossed & (rad > r_out)

        # --- dense shaping: potential on distance to the next gate ------------
        dist = np.linalg.norm(c - self.pos, axis=-1)
        progress = self.prev_dist - dist
        # a gate change makes the potential discontinuous; zero it on those steps
        progress = np.where(crossed, 0.0, progress)

        speed = np.linalg.norm(self.vel, axis=-1)
        to_gate = (c - self.pos) / np.maximum(dist, 1e-6)[:, None]
        closing = np.einsum("ej,ej->e", self.vel, to_gate)

        upright = R_bw[:, 2, 2]
        spin = np.linalg.norm(self.omega_slow, axis=-1)
        jerk = np.mean((action - self.prev_action) ** 2, axis=-1)
        p_rel = power / self.p_hover

        r = (cfg.w_progress * progress
             + cfg.w_speed * np.clip(closing, 0, None)
             - cfg.w_energy * p_rel
             - cfg.w_jerk * jerk
             - cfg.w_spin * (spin / 20.0) ** 2
             + cfg.w_upright * upright
             + cfg.w_alive)

        r += np.where(passed, cfg.w_gate + cfg.w_center * (1.0 - rad / np.maximum(r_in, 1e-9)), 0.0)
        r += np.where(missed, cfg.r_miss, 0.0)

        # --- termination -------------------------------------------------------
        ground = self.pos[:, 2] < 0.008
        oob = (np.linalg.norm(self.pos[:, :2], axis=-1) > cfg.arena_radius) | (self.pos[:, 2] > cfg.ceiling)
        tumbling = spin > 90.0
        crashed = ground | oob | clipped | tumbling
        r = np.where(crashed, r + cfg.r_crash, r)

        advance = passed | missed
        self.next_gate = np.where(advance, self.next_gate + 1, self.next_gate)
        finished = self.next_gate >= cfg.n_gates
        r = np.where(finished, r + cfg.r_finish, r)

        terminated = crashed | finished
        ngi = np.minimum(self.next_gate, cfg.n_gates - 1)
        self.prev_dist = np.linalg.norm(self.gates.center[ar, ngi] - self.pos, axis=-1)

        info = dict(gate_passed=passed, gate_missed=missed, crashed=crashed,
                    hit_ground=ground, out_of_bounds=oob, hit_ring=clipped, tumbled=tumbling,
                    finished=finished, gates_done=self.next_gate.copy(),
                    speed=speed, power_rel=p_rel, upright=upright, spin=spin,
                    jerk=jerk, gate_radius_err=np.where(passed, rad / np.maximum(r_in, 1e-9), np.nan))
        return r, terminated, info

    # ------------------------------------------------------------- observation
    def _observe(self, reset_mask=None):
        E = self.cfg.n_envs
        ar = np.arange(E)
        R_bw = _quat_to_mat(self.quat)
        R_wb = np.transpose(R_bw, (0, 2, 1))

        v_body = np.einsum("eij,ej->ei", R_wb, self.vel)
        grav_body = np.einsum("eij,j->ei", R_wb, np.array([0.0, 0.0, -1.0]))

        g0 = np.minimum(self.next_gate, self.cfg.n_gates - 1)
        g1 = np.minimum(self.next_gate + 1, self.cfg.n_gates - 1)
        feats = []
        for g in (g0, g1):
            c = self.gates.center[ar, g]
            n = self.gates.normal[ar, g]
            rel = c - self.pos
            d = np.linalg.norm(rel, axis=-1, keepdims=True)
            feats.append(np.einsum("eij,ej->ei", R_wb, rel / np.maximum(d, 1e-6)))
            feats.append(np.log1p(d / 0.2))
            feats.append(np.einsum("eij,ej->ei", R_wb, n))

        ph = self.wpg.phase
        proprio = np.concatenate([
            self.omega_slow / 20.0,            # haltere: beat-averaged body rate
            grav_body,                         # ocelli + haltere: which way is down
            v_body / 0.5,                      # Johnston's organ: airspeed
            np.stack([np.sin(ph), np.cos(ph)], -1),   # wing campaniform: stroke phase
            self.prev_action,
            *feats,
        ], axis=-1).astype(np.float32)

        obs = {"proprio": proprio}
        if self.eye is not None:
            due = (self._vis_tick % self.cfg.vision_every == 0) or (reset_mask is not None)
            self._vis_tick += 1
            if due:
                head = self.wpg.controls[:, 12:14]
                lum = self.eye.render(self.pos, R_bw, self._gate_window(), head)
                self.last_lum = lum
                dt_v = self.dt * self.cfg.vision_every
                self.last_on, self.last_off = self.eye.photoreceptor_drive(lum, dt_v, reset_mask)
            obs["on"], obs["off"] = self.last_on, self.last_off
        return obs

    def _gate_window(self) -> "_GateArray":
        """The few gates that can plausibly be in view: the one just passed
        through the next few ahead. Ray-casting the whole course every frame is
        wasted work — the rest are behind the fly or beyond its visual range."""
        E, G = self.cfg.n_envs, self.cfg.n_gates
        K = min(self.cfg.visible_gates, G)
        base = np.clip(self.next_gate - 1, 0, max(G - K, 0))
        sel = base[:, None] + np.arange(K)[None, :]
        ar = np.arange(E)[:, None]
        v = _GateArray.__new__(_GateArray)
        v.center = self.gates.center[ar, sel]
        v.normal = self.gates.normal[ar, sel]
        v.r_in = self.gates.r_in[ar, sel]
        v.r_out = self.gates.r_out[ar, sel]
        return v

    # -------------------------------------------------------------- inspection
    def snapshot(self, i: int = 0) -> dict:
        """Everything the monitor needs about environment `i`, JSON-ready."""
        R_bw = _quat_to_mat(self.quat[i : i + 1])[0]
        ws = self.wpg.kinematics()
        return dict(
            pos=self.pos[i].tolist(), vel=self.vel[i].tolist(),
            quat=self.quat[i].tolist(), omega=self.omega_slow[i].tolist(),
            R=R_bw.ravel().tolist(),
            speed=float(np.linalg.norm(self.vel[i])),
            next_gate=int(self.next_gate[i]),
            course=self.courses[i].to_dict(),
            wing=dict(phase=float(self.wpg.phase[i]), freq=float(ws.freq[i]),
                      phi=ws.phi[i].tolist(), theta=ws.theta[i].tolist(),
                      alpha=ws.alpha[i].tolist(),
                      aoa=self.last_diag.get("aoa", np.zeros((1, 2)))[i].tolist()
                      if self.last_diag else [0, 0],
                      tip_speed=self.last_diag.get("tip_speed", np.zeros((1, 2)))[i].tolist()
                      if self.last_diag else [0, 0]),
            power_rel=float(self.last_diag["power"][i] / self.p_hover) if self.last_diag else 0.0,
            scope=np.roll(self._scope, -self._scope_i, axis=0).round(4).T.tolist(),
            luminance=self.last_lum[i].tolist() if self.n_columns else [],
        )

    @property
    def action_dim(self) -> int:
        return N_WING_CONTROLS


# ------------------------------------------------------------------ quaternion
def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _quat_integrate(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    wx, wy, wz = omega_body[..., 0], omega_body[..., 1], omega_body[..., 2]
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    dq = 0.5 * np.stack([
        -x * wx - y * wy - z * wz,
        w * wx + y * wz - z * wy,
        w * wy - x * wz + z * wx,
        w * wz + x * wy - y * wx,
    ], -1)
    q = q + dq * dt
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _quat_from_forward(d: np.ndarray) -> np.ndarray:
    d = d / np.linalg.norm(d)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(d, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up2 = np.cross(right, d)
    M = np.stack([d, right, up2], axis=-1)
    tr = np.trace(M)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s])
    else:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    return q / np.linalg.norm(q)