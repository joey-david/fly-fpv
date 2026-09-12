"""Compound-eye rendering on the connectome's retinotopic hex grid.

MaleCNS supplies `assignedOlHex1/2` for retinotopic columnar neurons. Those
coordinates determine which neurons share an ommatidial column. They do *not*
contain optical viewing directions, so the conversion from the regular hex grid
to the unit sphere remains a calibration model rather than measured per-facet
optics.

The default angular envelope follows the whole-eye measurements in Zhao et al.,
Nature 646, 135-142 (2025): each eye reaches slightly across the frontal midline,
about 155 degrees posteriorly, from roughly -70 degrees elevation to the dorsal
pole. This gives <20 degrees binocular overlap and about a 50 degree posterior
blind spot. The internal hex topology is the measured MaleCNS topology.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ON_TYPES = {"L1", "L5", "Mi1"}
OFF_TYPES = {"L2", "L3", "Tm1", "Tm2", "Tm9"}


@dataclass
class EyeConfig:
    # Per eye, azimuth is measured outward from forward. Negative values cross
    # the frontal midline; the left eye is mirrored. Zhao et al. 2025 report
    # <10 deg contralateral front, ~155 deg posterior, +90/-70 elevation.
    fov_azimuth: tuple[float, float] = (-8.0, 155.0)
    fov_elevation: tuple[float, float] = (-70.0, 90.0)
    acceptance_angle: float = 5.0
    contrast_tau: float = 0.020
    ground_period: float = 0.04
    max_range: float = 4.0


class CompoundEye:
    """Ray-cast a hoop course through a connectome-derived ommatidial lattice."""

    def __init__(self, hex_coords: dict[int, tuple[int, int, str]], n_envs: int,
                 cfg: EyeConfig | None = None):
        self.cfg = cfg or EyeConfig()
        self.n_envs = n_envs

        cols: dict[tuple[int, int, str], list[int]] = {}
        for nid, (h1, h2, side) in hex_coords.items():
            cols.setdefault((h1, h2, side), []).append(nid)
        self.columns = sorted(cols.keys())
        self.column_neurons = [np.array(cols[k], dtype=np.int64) for k in self.columns]
        self.n_columns = len(self.columns)

        self.col_side = np.array([1 if k[2] == "R" else -1 for k in self.columns])
        self.directions = self._hex_to_directions()

        self._on_rows, self._on_cols = [], []
        self._off_rows, self._off_cols = [], []
        self.prev_luminance = np.zeros((n_envs, self.n_columns), dtype=np.float32)
        self._initialised = np.zeros(n_envs, dtype=bool)

    def _hex_to_directions(self) -> np.ndarray:
        """Map the MaleCNS hex sheet smoothly onto the measured visual envelope.

        The correspondence of neighbouring columns is exact. Absolute angular
        positions are approximate because MaleCNS does not ship a per-ommatidium
        optical calibration. Do not interpret these directions as measured lens
        axes; replacing this function with a matched micro-CT eye map would be a
        scientifically cleaner future calibration.
        """
        h1 = np.array([k[0] for k in self.columns], dtype=np.float64)
        h2 = np.array([k[1] for k in self.columns], dtype=np.float64)
        x = h1 + 0.5 * h2
        y = (np.sqrt(3) / 2) * h2

        dirs = np.zeros((self.n_columns, 3))
        for side in (1, -1):
            mask = self.col_side == side
            if not mask.any():
                continue
            # A regular-grid mapping is intentionally used here: unlike an
            # arbitrary perspective camera, it preserves the measured neighbour
            # graph and does not pretend we know unprovided per-facet optics.
            u = _unit_range(x[mask])
            v = _unit_range(y[mask])
            az0, az1 = np.deg2rad(self.cfg.fov_azimuth)
            el0, el1 = np.deg2rad(self.cfg.fov_elevation)
            az = (az0 + u * (az1 - az0)) * side
            el = el0 + v * (el1 - el0)
            dirs[mask] = np.stack(
                [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1
            )
        return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    def render(self, pos: np.ndarray, R_bw: np.ndarray, gates, head: np.ndarray) -> np.ndarray:
        """Return luminance per ommatidium, shape ``(envs, columns)`` in [0, 1]."""
        E, C = pos.shape[0], self.n_columns
        R_head = _rot_yz(head[:, 0], head[:, 1])
        d_body = np.einsum("eij,cj->eci", R_head, self.directions)
        d = np.einsum("eij,ecj->eci", R_bw, d_body)

        lum = 0.55 + 0.45 * np.clip(d[..., 2], 0, 1)

        pz = pos[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_g = np.where(d[..., 2] < -1e-6, -pz / d[..., 2], np.inf)
        t_g = np.where((t_g > 0) & (t_g < self.cfg.max_range), t_g, np.inf)
        hit_g = np.isfinite(t_g)
        if hit_g.any():
            p = pos[:, None, :] + np.where(hit_g, t_g, 0.0)[..., None] * d
            k = 2 * np.pi / self.cfg.ground_period
            tex = 0.5 + 0.5 * np.sin(k * p[..., 0]) * np.sin(k * p[..., 1])
            blur = np.clip(
                self.cfg.ground_period
                / (np.deg2rad(self.cfg.acceptance_angle) * np.maximum(t_g, 1e-6) * 4),
                0,
                1,
            )
            ground = 0.25 + 0.45 * (tex * blur + 0.5 * (1 - blur))
            lum = np.where(hit_g, ground, lum)
            t_best = np.where(hit_g, t_g, np.inf)
        else:
            t_best = np.full((E, C), np.inf)

        for gi in range(gates.center.shape[1]):
            c = gates.center[:, gi, :]
            nrm = gates.normal[:, gi, :]
            denom = np.einsum("ecj,ej->ec", d, nrm)
            ok = np.abs(denom) > 1e-6
            num = np.einsum("ej,ej->e", c - pos, nrm)[:, None]
            t = np.where(ok, num / np.where(ok, denom, 1.0), np.inf)
            t = np.where((t > 0) & (t < self.cfg.max_range), t, np.inf)
            hp = pos[:, None, :] + np.where(np.isfinite(t), t, 0.0)[..., None] * d
            rad = np.linalg.norm(hp - c[:, None, :], axis=-1)
            on_ring = (
                np.isfinite(t)
                & (rad >= gates.r_in[:, gi : gi + 1])
                & (rad <= gates.r_out[:, gi : gi + 1])
            )
            closer = on_ring & (t < t_best)
            if closer.any():
                rel = hp - c[:, None, :]
                e1 = _perp_batch(nrm)
                e2 = np.cross(nrm, e1)
                ang = np.arctan2(
                    np.einsum("ecj,ej->ec", rel, e2),
                    np.einsum("ecj,ej->ec", rel, e1),
                )
                seg = 0.15 + 0.70 * (np.sin(8 * ang) > 0)
                lum = np.where(closer, seg, lum)
                t_best = np.where(closer, t, t_best)

        return lum.astype(np.float32)

    def photoreceptor_drive(self, luminance: np.ndarray, dt: float, reset_mask=None):
        """Compute the policy's ON/OFF channels from rendered luminance."""
        if reset_mask is not None and reset_mask.any():
            self.prev_luminance[reset_mask] = luminance[reset_mask]
        a = dt / (self.cfg.contrast_tau + dt)
        delta = (luminance - self.prev_luminance) / max(dt, 1e-6)
        self.prev_luminance += a * (luminance - self.prev_luminance)

        contrast = np.tanh(delta * self.cfg.contrast_tau)
        static = luminance - 0.5
        on = np.maximum(contrast, 0) + 0.3 * np.maximum(static, 0)
        off = np.maximum(-contrast, 0) + 0.3 * np.maximum(-static, 0)
        return on.astype(np.float32), off.astype(np.float32)

    def scatter_to_neurons(self, on, off, n_neurons: int, neuron_type: np.ndarray) -> np.ndarray:
        """Place ON/OFF channels onto their retinotopic neurons."""
        drive = np.zeros((on.shape[0], n_neurons), dtype=np.float32)
        for ci, rows in enumerate(self.column_neurons):
            if not len(rows):
                continue
            types = neuron_type[rows]
            is_on = np.isin(types, list(ON_TYPES))
            if is_on.any():
                drive[:, rows[is_on]] = on[:, ci : ci + 1]
            is_off = ~is_on
            if is_off.any():
                drive[:, rows[is_off]] = off[:, ci : ci + 1]
        return drive


def _unit_range(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(), a.max()
    return (a - lo) / max(hi - lo, 1e-9)


def _perp_batch(n: np.ndarray) -> np.ndarray:
    a = np.where(
        np.abs(n[:, 2:3]) < 0.9,
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    v = np.cross(n, a)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _rot_yz(pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    z, o = np.zeros_like(cp), np.ones_like(cp)
    Ry = np.stack(
        [
            np.stack([cp, z, sp], -1),
            np.stack([z, o, z], -1),
            np.stack([-sp, z, cp], -1),
        ],
        -2,
    )
    Rz = np.stack(
        [
            np.stack([cy, -sy, z], -1),
            np.stack([sy, cy, z], -1),
            np.stack([z, z, o], -1),
        ],
        -2,
    )
    return Rz @ Ry
