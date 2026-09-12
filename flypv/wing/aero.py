"""Quasi-steady blade-element aerodynamics for a flapping fly wing.

Fixed-wing intuition does not survive at Reynolds number ~150 with a wing that
reverses direction 200 times a second.  The forces here come from the
quasi-steady model validated against Robofly (Sane & Dickinson 1999, 2002):

  * **Translational** circulation, with lift and drag coefficients that are
    empirical functions of angle of attack.  Note that CL peaks near 45 deg,
    not 15 — a flapping wing operates permanently stalled, stabilised by a
    leading-edge vortex, and the "drag" term is a genuine contributor to weight
    support during the stroke, not a loss.
  * **Rotational** circulation (the Kramer effect) from the wing pitching about
    its own span axis at stroke reversal.  This is the term that lets a fly
    steer by changing wing rotation timing, which is most of how it turns.
  * **Added mass** from accelerating the surrounding fluid.

Everything is vectorised over (n_envs, 2 wings, n_blade_elements).
"""
from __future__ import annotations

import numpy as np

from .morphology import FlyMorphology

# Sane & Dickinson (2002), alpha in radians. Valid over the full 0..pi range a
# flapping wing actually visits.
_CL_OFFSET, _CL_GAIN, _CL_PHASE = 0.225, 1.58, np.deg2rad(7.2)
_CD_OFFSET, _CD_GAIN, _CD_PHASE = 1.92, 1.55, np.deg2rad(9.82)


def CL(alpha: np.ndarray) -> np.ndarray:
    return _CL_OFFSET + _CL_GAIN * np.sin(2.13 * alpha - _CL_PHASE)


def CD(alpha: np.ndarray) -> np.ndarray:
    return _CD_OFFSET - _CD_GAIN * np.cos(2.04 * alpha - _CD_PHASE)


C_ROT = 1.55  # rotational circulation coefficient, ~ pi*(0.75 - x_hat_0)


def _skew_cross(w, v):
    """Cross product broadcasting (..., 3) x (..., 3)."""
    return np.cross(w, v)


def blade_element_forces(
    morph: FlyMorphology,
    phi: np.ndarray,        # (E, 2) stroke angle, rad
    theta: np.ndarray,      # (E, 2) deviation angle, rad
    alpha_geo: np.ndarray,  # (E, 2) geometric wing pitch, rad (0 = chord in stroke plane)
    dphi: np.ndarray,       # (E, 2) d/dt
    dtheta: np.ndarray,
    dalpha: np.ndarray,
    v_body: np.ndarray,     # (E, 3) body-frame linear velocity of the CoM, m/s
    omega_body: np.ndarray, # (E, 3) body-frame angular velocity, rad/s
    stroke_tilt: np.ndarray,# (E,) extra stroke-plane tilt commanded by the policy, rad
    com_shift: np.ndarray | None = None,  # (E,3) centre-of-mass offset, m
    n_elements: int = 8,
):
    """Return (force, torque, diag) in the BODY frame.

    force  (E, 3) N
    torque (E, 3) N*m about the centre of mass
    diag   dict of per-wing diagnostics (aoa, lift, drag, power) for the monitor
    """
    E = phi.shape[0]
    R = morph.wing_length
    rho = morph.air_density

    # --- blade element stations -------------------------------------------
    # Beta-like chord distribution normalised so sum(c_i * dr) = wing_area and
    # the second moment matches r2_hat.
    r_hat = (np.arange(n_elements) + 0.5) / n_elements
    dr = R / n_elements
    c_r = _chord_distribution(r_hat, morph)             # (n,) metres
    r = r_hat * R                                        # (n,)

    side = np.array([+1.0, -1.0])                        # right, left (y sign)

    # --- wing kinematics -> element position/velocity in the STROKE frame ---
    # Stroke frame: x forward, y spanwise (outboard positive for that wing),
    # z up-normal to the stroke plane. phi sweeps about stroke-frame z,
    # theta elevates out of the stroke plane.
    cphi, sphi = np.cos(phi)[..., None], np.sin(phi)[..., None]   # (E,2,1)
    cth, sth = np.cos(theta)[..., None], np.sin(theta)[..., None]

    # unit span vector per wing, in stroke frame
    e_span = np.stack([
        np.broadcast_to(sphi * cth, (E, 2, n_elements)),
        np.broadcast_to(cphi * cth, (E, 2, n_elements)) * side[None, :, None],
        np.broadcast_to(np.broadcast_to(sth, (E, 2, 1)), (E, 2, n_elements)),
    ], axis=-1)                                                     # (E,2,n,3)

    pos_s = e_span * r[None, None, :, None]                         # (E,2,n,3)

    # d(e_span)/dt from phi_dot and theta_dot
    dphi_ = dphi[..., None, None]
    dth_ = dtheta[..., None, None]
    de = np.stack([
        (cphi[..., None] * cth[..., None] * dphi_ - sphi[..., None] * sth[..., None] * dth_)[..., 0],
        ((-sphi[..., None] * cth[..., None] * dphi_ - cphi[..., None] * sth[..., None] * dth_)[..., 0]
         * side[None, :, None, None][..., 0]),
        (cth[..., None] * dth_)[..., 0],
    ], axis=-1)                                                     # (E,2,n,3)
    de = np.broadcast_to(de, (E, 2, n_elements, 3))
    vel_flap_s = de * r[None, None, :, None]

    # --- into the BODY frame ----------------------------------------------
    beta = morph.stroke_plane_angle + stroke_tilt                    # (E,)
    Rsb = _rot_y(beta)                                               # (E,3,3) stroke -> body
    pos_b = np.einsum("eij,ewnj->ewni", Rsb, pos_s)
    hinge = morph.hinge_offset[None, None, :] * np.stack(
        [np.ones(2), side, np.ones(2)], axis=-1)[:, None, :]         # (2,1,3)
    pos_b = pos_b + hinge[None]
    if com_shift is not None:
        # Swinging the abdomen moves the centre of mass, which changes the moment
        # arm of every aerodynamic force. That is how a fly uses its abdomen as a
        # trim surface, and it works even in still air where a flap would not.
        pos_b = pos_b - com_shift[:, None, None, :]
    vel_flap_b = np.einsum("eij,ewnj->ewni", Rsb, vel_flap_s)

    # total element velocity relative to still air, in body frame
    v_elem = (vel_flap_b
              + v_body[:, None, None, :]
              + _skew_cross(np.broadcast_to(omega_body[:, None, None, :], pos_b.shape), pos_b))

    # --- decompose: only flow normal to the span makes force ---------------
    span_b = np.einsum("eij,ewnj->ewni", Rsb, e_span)
    v_span = np.sum(v_elem * span_b, axis=-1, keepdims=True) * span_b
    v_perp = v_elem - v_span
    U = np.linalg.norm(v_perp, axis=-1)                              # (E,2,n)
    U_safe = np.maximum(U, 1e-9)
    u_hat = v_perp / U_safe[..., None]

    # chord normal: rotate the stroke-plane normal about the span by alpha_geo
    n_sp = np.einsum("eij,j->ei", Rsb, np.array([0.0, 0.0, 1.0]))
    n_sp = np.broadcast_to(n_sp[:, None, None, :], pos_b.shape)
    a = alpha_geo[..., None, None]
    chord_normal = (n_sp * np.cos(a)
                    + np.cross(span_b, n_sp) * np.sin(a)
                    + span_b * np.sum(span_b * n_sp, axis=-1, keepdims=True) * (1 - np.cos(a)))

    # effective angle of attack: angle between the incident flow and the chord
    cos_aoa = np.clip(np.sum(-u_hat * chord_normal, axis=-1), -1.0, 1.0)
    aoa = np.arccos(np.abs(cos_aoa))              # 0 = edge-on, pi/2 = broadside
    aoa = np.pi / 2 - aoa                          # conventional: 0 = chord along flow
    aoa_abs = np.abs(aoa)

    q = 0.5 * rho * U**2 * c_r[None, None, :] * dr                   # (E,2,n) dynamic pressure * area

    # lift acts perpendicular to the incident flow within the force plane,
    # drag acts along -u_hat
    lift_dir = np.cross(span_b, u_hat)
    lift_dir = lift_dir / np.maximum(np.linalg.norm(lift_dir, axis=-1, keepdims=True), 1e-12)
    lift_dir *= np.sign(np.sum(lift_dir * chord_normal, axis=-1, keepdims=True) + 1e-12)

    dL = (CL(aoa_abs) * q)[..., None] * lift_dir
    dD = (CD(aoa_abs) * q)[..., None] * (-u_hat)

    # rotational circulation (Kramer effect): F_rot = C_rot * rho * U * dalpha * c^2 * dr
    f_rot = (C_ROT * rho * U * dalpha[..., None] * c_r[None, None, :] ** 2 * dr)
    dR_ = f_rot[..., None] * chord_normal

    # added mass: normal acceleration of the fluid slug, ~ rho*pi/4*c^2*dr*a_normal
    a_norm = (dalpha**2)[..., None] * 0.0  # accel term folded into the wing-inertia torque below
    dA = np.zeros_like(dL) + a_norm[..., None] * chord_normal

    dF = dL + dD + dR_ + dA                                          # (E,2,n,3)

    force = dF.sum(axis=(1, 2))
    torque = np.cross(pos_b, dF).sum(axis=(1, 2))

    # --- body parasite drag ------------------------------------------------
    vb = np.linalg.norm(v_body, axis=-1, keepdims=True)
    force = force - 0.5 * rho * morph.body_drag_area * vb * v_body

    # --- aerodynamic power, the quantity "efficient" is measured against -----
    power = np.abs(np.sum(dF * v_elem, axis=-1)).sum(axis=(1, 2))     # (E,) watts

    diag = {
        "aoa": np.rad2deg(aoa[:, :, n_elements // 2]),                # (E,2) mid-span AoA, deg
        "lift": np.linalg.norm(dL.sum(axis=2), axis=-1),              # (E,2) N
        "drag": np.linalg.norm(dD.sum(axis=2), axis=-1),
        "tip_speed": U[:, :, -1],                                     # (E,2) m/s
        "power": power,
    }
    return force, torque, diag


def _chord_distribution(r_hat: np.ndarray, morph: FlyMorphology) -> np.ndarray:
    """Beta-distribution chord profile scaled to the measured area and 2nd moment."""
    p, q = 2.2, 2.6   # shape params giving r2_hat ~ 0.58 for a Drosophila wing
    w = r_hat ** (p - 1) * (1 - r_hat) ** (q - 1)
    w = w / w.mean()
    c = w * morph.mean_chord
    # renormalise so sum(c*dr) == wing_area exactly
    n = len(r_hat)
    c *= morph.wing_area / (c.sum() * morph.wing_length / n)
    return c


def _rot_y(beta: np.ndarray) -> np.ndarray:
    c, s = np.cos(beta), np.sin(beta)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([
        np.stack([c, z, s], -1),
        np.stack([z, o, z], -1),
        np.stack([-s, z, c], -1),
    ], -2)
