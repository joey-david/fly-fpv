"""A hand-written stabilising pilot, used for two jobs.

First, as a check on the environment: if no controller can fly the course, a
learning agent failing tells you nothing. Second, as the expert for the
behaviour-cloning warm start — an unstable plant gives a from-scratch policy
almost no usable gradient before it crashes, so the same two-stage recipe
(imitate a simple controller, then improve with PPO) that connectome-policy work
uses applies here.

Nothing is tuned by hand in the dark. Each channel's authority was measured
directly from the blade-element model by perturbing it and integrating over a
wingbeat; `AUTHORITY` below is that measurement, in body weights of force and in
torque non-dimensionalised by weight x wing length. The controller asks for a
force and a torque in physical units and divides.

Structure is the standard geometric attitude controller, which suits a fly well
because a hovering flapper is a thrust-vectoring body: pick the acceleration you
want, point the lift axis along it, and hold the nose on the gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..wing import DROSOPHILA, WING_CONTROLS

IX = {c: i for i, c in enumerate(WING_CONTROLS)}

#: Measured response per unit of control, at hover trim.
#: Torques are normalised by (weight x wing length), forces by weight.
AUTHORITY = {
    "roll_per_amp_diff": 0.173,      # (amp_R - amp_L), full units
    "pitch_per_offset": -0.211,      # collective stroke offset, both sides
    "yaw_per_aoa_diff": 0.329,       # (aoa_down_R - aoa_down_L)
    "lift_per_amp_sum": 0.760,       # collective amplitude
    "thrust_per_tilt": 0.512,        # stroke-plane tilt
}


@dataclass
class ReflexGains:
    # Cascaded attitude loop: angle error sets a desired body *rate*, which is
    # then clamped, and an inner loop drives the rate. A single-stage PD on angle
    # asks for slews of thousands of deg/s when a gate is far off the nose, and
    # the fly tumbles executing its own command.
    k_angle: float = 9.0            # 1/s, angle error -> desired body rate
    k_rate: float = 260.0           # 1/s, rate error -> angular acceleration
    rate_max: float = 35.0          # rad/s, about 2000 deg/s, a real fly's saccade peak
    k_angle_yaw: float = 6.0
    rate_max_yaw: float = 25.0
    # guidance loop, deliberately an order of magnitude slower
    w_pos: float = 7.0
    zeta_pos: float = 1.1
    target_speed: float = 0.50      # m/s along the line to the next gate
    max_tilt: float = 0.9           # rad, how far it will tip the lift axis
    lead: float = 0.07              # m past the gate plane to aim for


class ReflexPilot:
    """Vectorised geometric attitude-and-guidance controller over the 15 controls."""

    def __init__(self, gains: ReflexGains | None = None, noise: float = 0.0,
                 morph=DROSOPHILA, seed: int = 0):
        self.g = gains or ReflexGains()
        self.m = morph
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        # angular acceleration -> normalised torque, per axis
        self.inertia_norm = morph.inertia / (morph.weight * morph.wing_length)

    def __call__(self, env) -> np.ndarray:
        g, G = self.g, self.m.gravity
        E = env.cfg.n_envs
        u = np.zeros((E, len(WING_CONTROLS)))

        R = _quat_to_mat(env.quat)                       # body -> world
        fwd, right, up = R[:, :, 0], R[:, :, 1], R[:, :, 2]

        ar = np.arange(E)
        gi = np.minimum(env.next_gate, env.cfg.n_gates - 1)
        normal = env.gates.normal[ar, gi]
        target = env.gates.center[ar, gi] + g.lead * normal

        to = target - env.pos
        dist = np.linalg.norm(to, axis=-1, keepdims=True)
        dir_ = to / np.maximum(dist, 1e-6)

        # --- guidance: the acceleration that would fly us to the gate ----------
        v_des = dir_ * g.target_speed
        kp_p, kd_p = g.w_pos ** 2, 2 * g.zeta_pos * g.w_pos
        a_des = kp_p * to * 0.25 + kd_p * (v_des - env.vel)
        a_des[:, 2] += G                                  # hold weight
        # do not ask for more tilt than the fly can hold
        horiz = np.linalg.norm(a_des[:, :2], axis=-1)
        lim = np.tan(g.max_tilt) * np.maximum(a_des[:, 2], 0.3 * G)
        scale = np.minimum(1.0, lim / np.maximum(horiz, 1e-6))
        a_des[:, :2] *= scale[:, None]

        z_des = a_des / np.maximum(np.linalg.norm(a_des, axis=-1, keepdims=True), 1e-9)

        # --- attitude error: rotation taking the lift axis onto z_des ----------
        e_world = np.cross(up, z_des)
        sin_ang = np.linalg.norm(e_world, axis=-1, keepdims=True)
        cos_ang = np.einsum("ej,ej->e", up, z_des)[:, None]
        ang = np.arctan2(sin_ang, cos_ang)
        axis = e_world / np.maximum(sin_ang, 1e-9)
        err_w = axis * ang
        err_b = np.einsum("eji,ej->ei", R, err_w)          # into the body frame

        # yaw: bring the nose onto the gate, independent of the lift axis
        dir_b = np.einsum("eji,ej->ei", R, dir_)
        yaw_err = np.arctan2(dir_b[:, 1], np.maximum(dir_b[:, 0], 1e-3))

        om = env.omega_slow
        rate_des = np.stack([
            np.clip(g.k_angle * err_b[:, 0], -g.rate_max, g.rate_max),
            np.clip(g.k_angle * err_b[:, 1], -g.rate_max, g.rate_max),
            np.clip(g.k_angle_yaw * yaw_err, -g.rate_max_yaw, g.rate_max_yaw),
        ], -1)
        alpha = g.k_rate * (rate_des - om)
        torque = alpha * self.inertia_norm[None, :]        # normalised units

        # --- invert the measured control map -----------------------------------
        A = AUTHORITY
        amp_diff = torque[:, 0] / A["roll_per_amp_diff"]
        offset = torque[:, 1] / A["pitch_per_offset"]
        aoa_diff = torque[:, 2] / A["yaw_per_aoa_diff"]

        lift_need = np.einsum("ej,ej->e", a_des, up) / G    # in body weights
        amp_sum = (lift_need - 1.0) / A["lift_per_amp_sum"]
        thrust_need = np.einsum("ej,ej->e", a_des, fwd) / G
        tilt = thrust_need / A["thrust_per_tilt"]

        u[:, IX["amp_R"]] = amp_sum + 0.5 * amp_diff
        u[:, IX["amp_L"]] = amp_sum - 0.5 * amp_diff
        u[:, IX["offset_R"]] = 0.5 * offset
        u[:, IX["offset_L"]] = 0.5 * offset
        u[:, IX["aoa_down_R"]] = 0.5 * aoa_diff
        u[:, IX["aoa_down_L"]] = -0.5 * aoa_diff
        u[:, IX["stroke_tilt"]] = tilt
        u[:, IX["abdomen_pitch"]] = np.clip(0.5 * offset, -1, 1)
        # gaze: hold the horizon with the head so vision stays interpretable
        u[:, IX["head_pitch"]] = np.clip(-1.2 * np.arcsin(np.clip(fwd[:, 2], -1, 1)), -1, 1)
        u[:, IX["head_yaw"]] = np.clip(0.8 * yaw_err, -1, 1)

        if self.noise:
            u += self.rng.normal(0, self.noise, u.shape)
        return np.clip(u, -1.0, 1.0)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    from ..env.hoops import _quat_to_mat as f
    return f(q)
