"""Quasi-steady blade-element aerodynamics for a flapping fly wing.

The model includes translational lift/drag using the Sane & Dickinson fits,
rotational circulation around stroke reversal, body parasite drag, and the
resulting force/torque and aerodynamic power. It intentionally does not claim an
added-mass term: the previous implementation contained a zero-valued placeholder
for one, which has been removed.

Everything is vectorized over (n_envs, 2 wings, n_blade_elements).
"""
from __future__ import annotations

import numpy as np

from .morphology import FlyMorphology

# Sane & Dickinson (2002), alpha in radians.
_CL_OFFSET, _CL_GAIN, _CL_PHASE = 0.225, 1.58, np.deg2rad(7.2)
_CD_OFFSET, _CD_GAIN, _CD_PHASE = 1.92, 1.55, np.deg2rad(9.82)
C_ROT = 1.55


def CL(alpha: np.ndarray) -> np.ndarray:
    return _CL_OFFSET + _CL_GAIN * np.sin(2.13 * alpha - _CL_PHASE)


def CD(alpha: np.ndarray) -> np.ndarray:
    return _CD_OFFSET - _CD_GAIN * np.cos(2.04 * alpha - _CD_PHASE)


def blade_element_forces(
    morph: FlyMorphology,
    phi: np.ndarray,
    theta: np.ndarray,
    alpha_geo: np.ndarray,
    dphi: np.ndarray,
    dtheta: np.ndarray,
    dalpha: np.ndarray,
    v_body: np.ndarray,
    omega_body: np.ndarray,
    stroke_tilt: np.ndarray,
    com_shift: np.ndarray | None = None,
    n_elements: int = 8,
):
    """Return body-frame force, torque, and per-wing diagnostics.

    Angles have shape (E, 2); body velocity/rate have shape (E, 3). The force
    and torque outputs have shape (E, 3).
    """
    E = phi.shape[0]
    R = morph.wing_length
    rho = morph.air_density

    r_hat = (np.arange(n_elements) + 0.5) / n_elements
    dr = R / n_elements
    c_r = _chord_distribution(r_hat, morph)
    r = r_hat * R
    side = np.array([+1.0, -1.0])

    cphi, sphi = np.cos(phi)[..., None], np.sin(phi)[..., None]
    cth, sth = np.cos(theta)[..., None], np.sin(theta)[..., None]

    # Span direction in the stroke frame. Right/left differ by the sign of y.
    e_span = np.stack(
        [
            np.broadcast_to(sphi * cth, (E, 2, n_elements)),
            np.broadcast_to(cphi * cth, (E, 2, n_elements)) * side[None, :, None],
            np.broadcast_to(sth, (E, 2, n_elements)),
        ],
        axis=-1,
    )
    pos_s = e_span * r[None, None, :, None]

    # Time derivative of span direction from stroke and deviation rates.
    dphi_ = dphi[..., None]
    dtheta_ = dtheta[..., None]
    de_x = cphi * cth * dphi_ - sphi * sth * dtheta_
    de_y = (-sphi * cth * dphi_ - cphi * sth * dtheta_) * side[None, :, None]
    de_z = cth * dtheta_
    de = np.stack(
        [
            np.broadcast_to(de_x, (E, 2, n_elements)),
            np.broadcast_to(de_y, (E, 2, n_elements)),
            np.broadcast_to(de_z, (E, 2, n_elements)),
        ],
        axis=-1,
    )
    vel_flap_s = de * r[None, None, :, None]

    beta = morph.stroke_plane_angle + stroke_tilt
    Rsb = _rot_y(beta)
    pos_b = np.einsum("eij,ewnj->ewni", Rsb, pos_s)
    hinge = np.stack(
        [
            np.full(2, morph.hinge_offset[0]),
            side * abs(morph.hinge_offset[1]),
            np.full(2, morph.hinge_offset[2]),
        ],
        axis=-1,
    )
    pos_b += hinge[None, :, None, :]
    if com_shift is not None:
        pos_b -= com_shift[:, None, None, :]

    vel_flap_b = np.einsum("eij,ewnj->ewni", Rsb, vel_flap_s)
    v_elem = (
        vel_flap_b
        + v_body[:, None, None, :]
        + np.cross(np.broadcast_to(omega_body[:, None, None, :], pos_b.shape), pos_b)
    )

    span_b = np.einsum("eij,ewnj->ewni", Rsb, e_span)
    v_span = np.sum(v_elem * span_b, axis=-1, keepdims=True) * span_b
    v_perp = v_elem - v_span
    U = np.linalg.norm(v_perp, axis=-1)
    u_hat = v_perp / np.maximum(U[..., None], 1e-9)

    # Wing surface normal after pitching the chord around the span axis.
    n_sp = np.einsum("eij,j->ei", Rsb, np.array([0.0, 0.0, 1.0]))
    n_sp = np.broadcast_to(n_sp[:, None, None, :], pos_b.shape)
    a = alpha_geo[..., None, None]
    chord_normal = (
        n_sp * np.cos(a)
        + np.cross(span_b, n_sp) * np.sin(a)
        + span_b
        * np.sum(span_b * n_sp, axis=-1, keepdims=True)
        * (1 - np.cos(a))
    )

    cos_aoa = np.clip(np.sum(-u_hat * chord_normal, axis=-1), -1.0, 1.0)
    aoa = np.pi / 2 - np.arccos(np.abs(cos_aoa))
    aoa_abs = np.abs(aoa)
    q = 0.5 * rho * U**2 * c_r[None, None, :] * dr

    lift_dir = np.cross(span_b, u_hat)
    lift_dir /= np.maximum(np.linalg.norm(lift_dir, axis=-1, keepdims=True), 1e-12)
    lift_dir *= np.sign(
        np.sum(lift_dir * chord_normal, axis=-1, keepdims=True) + 1e-12
    )

    dL = (CL(aoa_abs) * q)[..., None] * lift_dir
    dD = (CD(aoa_abs) * q)[..., None] * (-u_hat)

    # Rotational circulation (Kramer effect) around stroke reversal.
    f_rot = C_ROT * rho * U * dalpha[..., None] * c_r[None, None, :] ** 2 * dr
    dR = f_rot[..., None] * chord_normal
    dF = dL + dD + dR

    force = dF.sum(axis=(1, 2))
    torque = np.cross(pos_b, dF).sum(axis=(1, 2))

    speed = np.linalg.norm(v_body, axis=-1, keepdims=True)
    force -= 0.5 * rho * morph.body_drag_area * speed * v_body

    power = np.abs(np.sum(dF * v_elem, axis=-1)).sum(axis=(1, 2))
    diag = {
        "aoa": np.rad2deg(aoa[:, :, n_elements // 2]),
        "lift": np.linalg.norm(dL.sum(axis=2), axis=-1),
        "drag": np.linalg.norm(dD.sum(axis=2), axis=-1),
        "tip_speed": U[:, :, -1],
        "power": power,
    }
    return force, torque, diag


def _chord_distribution(r_hat: np.ndarray, morph: FlyMorphology) -> np.ndarray:
    """Smooth Drosophila-like chord profile normalized to measured wing area."""
    p, q = 2.2, 2.6
    shape = r_hat ** (p - 1) * (1 - r_hat) ** (q - 1)
    shape /= shape.mean()
    chord = shape * morph.mean_chord
    n = len(r_hat)
    chord *= morph.wing_area / (chord.sum() * morph.wing_length / n)
    return chord


def _rot_y(beta: np.ndarray) -> np.ndarray:
    c, s = np.cos(beta), np.sin(beta)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack(
        [
            np.stack([c, z, s], -1),
            np.stack([z, o, z], -1),
            np.stack([-s, z, c], -1),
        ],
        -2,
    )
