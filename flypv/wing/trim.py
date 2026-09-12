"""Solve for the wing kinematics that actually hold a fly still in the air.

Nominal textbook values — 140 deg stroke amplitude, 218 Hz, 45 deg angle of
attack, a -15 deg stroke plane — produce about 0.95 body weights of lift and,
more importantly, a residual nose-up pitch torque of order 2000 rad/s^2. A fly
released with those kinematics tumbles in about 30 milliseconds.

That residual is not physics telling us flies cannot hover; it is the model
missing the fly's *anatomical* trim. Where the wing hinge sits relative to the
centre of mass, and where the mean stroke angle sits within the stroke, are what
set the pitch balance, and a real fly's are arranged so that its neutral gait is
close to equilibrium.

So we solve for them: find the mean stroke offset and amplitude scale for which
the beat-averaged vertical force equals body weight and the beat-averaged pitch
torque is zero. The result is cached on the morphology and becomes the zero
point of the control vector, so "do nothing" means "hover" and the policy spends
its capacity on flying rather than on rediscovering trim.

What this deliberately does *not* remove is the instability. Hovering flight is
still a saddle: a small pitch disturbance grows. The fly has to actively
stabilise, which is exactly the problem the haltere-to-steering-muscle circuitry
in the connectome exists to solve.
"""
from __future__ import annotations

import numpy as np

from .aero import blade_element_forces
from .morphology import FlyMorphology

_CACHE: dict[int, tuple[float, float]] = {}


def _cycle_average(morph: FlyMorphology, offset: float, amp_scale: float,
                   tilt: float = 0.0, n: int = 96):
    """Beat-averaged (Fx, Fz, pitch torque) for a symmetric stroke, no body motion."""
    from .wpg import ALPHA_MID, _FLIP_SHARPNESS

    p = np.linspace(0, 2 * np.pi, n, endpoint=False)
    amp = morph.amplitude_nominal * amp_scale
    w = 2 * np.pi * morph.f_nominal

    sp, cp = np.sin(p)[:, None], np.cos(p)[:, None]
    phi = offset + 0.5 * amp * sp * np.ones((1, 2))
    dphi = 0.5 * amp * cp * w * np.ones((1, 2))
    theta = np.zeros((n, 2))
    dtheta = np.zeros((n, 2))
    s = np.tanh(_FLIP_SHARPNESS * cp)
    ds = (1 - s**2) * _FLIP_SHARPNESS * (-sp) * w
    down = 0.5 * (1 + s)
    alpha = (ALPHA_MID * down - ALPHA_MID * (1 - down)) * np.ones((1, 2))
    dalpha = (ds * ALPHA_MID) * np.ones((1, 2))

    F, T, _ = blade_element_forces(
        morph, phi, theta, alpha, dphi, dtheta, dalpha,
        np.zeros((n, 3)), np.zeros((n, 3)), np.full(n, tilt))
    return float(F[:, 0].mean()), float(F[:, 2].mean()), float(T[:, 1].mean())


def solve_trim(morph: FlyMorphology, verbose: bool = False) -> tuple[float, float, float]:
    """Return (stroke_offset, amplitude_scale, stroke_tilt) that hover-trims this fly."""
    key = id(morph)
    if key in _CACHE:
        return _CACHE[key]

    def residual(v):
        Fx, Fz, My = _cycle_average(morph, v[0], v[1], v[2])
        return np.array([My / (morph.weight * morph.wing_length),
                         Fz / morph.weight - 1.0,
                         Fx / morph.weight])

    x = np.array([0.0, 1.0, 0.0])      # stroke offset, amplitude scale, stroke tilt
    for _ in range(80):
        res = residual(x)
        if np.abs(res).max() < 1e-5:
            break
        J = np.zeros((3, 3))
        for j in range(3):
            xp = x.copy(); xp[j] += 1e-3
            J[:, j] = (residual(xp) - res) / 1e-3
        try:
            step = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            break
        x = x + np.clip(step, -0.25, 0.25)
        x[1] = np.clip(x[1], 0.4, 2.0)
        x[0] = np.clip(x[0], -1.0, 1.0)
        x[2] = np.clip(x[2], -1.0, 1.0)

    Fx, Fz, My = _cycle_average(morph, x[0], x[1], x[2])
    if verbose:
        print(f"[trim] stroke offset {np.rad2deg(x[0]):+.2f} deg, amplitude "
              f"{np.rad2deg(morph.amplitude_nominal * x[1]):.1f} deg, stroke plane "
              f"{np.rad2deg(morph.stroke_plane_angle + x[2]):+.2f} deg -> "
              f"lift {Fz / morph.weight:.4f} W, drift {Fx / morph.weight:+.1e} W, "
              f"pitch {My / (morph.weight * morph.wing_length):+.1e}")
    _CACHE[key] = (float(x[0]), float(x[1]), float(x[2]))
    return _CACHE[key]
