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

Each ommatidium integrates over a finite angular footprint rather than sampling
a single center ray. Gate coverage is computed analytically from the overlap of
that footprint with the projected annulus, avoiding expensive multi-ray
supersampling while still antialiasing sub-ommatidial objects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ON_TYPES = {"L1", "L5", "Mi1"}
OFF_TYPES = {"L2", "L3", "Tm1", "Tm2", "Tm9"}

# Axial-neighbour offsets for the (h1, h2) lattice used below.
_HEX_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
# Equal-area disk radius / nearest-neighbour spacing for a hexagonal Voronoi cell.
_HEX_TO_DISK = np.sqrt(np.sqrt(3.0) / (2.0 * np.pi))


@dataclass
class EyeConfig:
    # Per eye, azimuth is measured outward from forward. Negative values cross
    # the frontal midline; the left eye is mirrored. Zhao et al. 2025 report
    # <10 deg contralateral front, ~155 deg posterior, +90/-70 elevation.
    fov_azimuth: tuple[float, float] = (-8.0, 155.0)
    fov_elevation: tuple[float, float] = (-70.0, 90.0)
    # Fallback full angular diameter for isolated/border columns. Interior
    # columns derive their footprint from the actual mapped hex-neighbour spacing.
    acceptance_angle: float = 5.0
    contrast_tau: float = 0.020
    ground_period: float = 0.04
    max_range: float = 4.0


class CompoundEye:
    """Render a hoop course through a connectome-derived ommatidial lattice."""

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
        self.footprint_half_angle = self._column_footprint_half_angles()
        self._tan_half_angle = np.tan(self.footprint_half_angle)[None, :]

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

    def _column_footprint_half_angles(self) -> np.ndarray:
        """Equivalent-disk half-angle for each displayed hexagonal visual cell.

        The visualizer draws one hex per MaleCNS column, so the most faithful
        antialiasing support is that cell's angular Voronoi area, not an arbitrary
        fixed supersampling kernel. For a regular hex lattice the Voronoi cell has
        area sqrt(3)/2 * s^2 for neighbour spacing s; `_HEX_TO_DISK * s` is the
        radius of a disk with the same area. Border columns fall back to half of
        `acceptance_angle`.
        """
        fallback = 0.5 * np.deg2rad(self.cfg.acceptance_angle)
        out = np.full(self.n_columns, fallback, dtype=np.float64)
        by_key = {k: i for i, k in enumerate(self.columns)}
        for i, (h1, h2, side) in enumerate(self.columns):
            angular_spacing = []
            di = self.directions[i]
            for dh1, dh2 in _HEX_NEIGHBOURS:
                j = by_key.get((h1 + dh1, h2 + dh2, side))
                if j is None:
                    continue
                angular_spacing.append(
                    np.arccos(np.clip(float(np.dot(di, self.directions[j])), -1.0, 1.0))
                )
            if angular_spacing:
                out[i] = _HEX_TO_DISK * float(np.median(angular_spacing))
        # Keep pathological edge mappings from creating vanishing or enormous
        # footprints while preserving normal local variation across the eye.
        return np.clip(out, np.deg2rad(0.5), np.deg2rad(6.0))

    def render(self, pos: np.ndarray, R_bw: np.ndarray, gates, head: np.ndarray) -> np.ndarray:
        """Return area-integrated luminance, shape ``(envs, columns)`` in [0, 1]."""
        E, C = pos.shape[0], self.n_columns
        R_head = _rot_yz(head[:, 0], head[:, 1])
        d_body = np.einsum("eij,cj->eci", R_head, self.directions)
        d = np.einsum("eij,ecj->eci", R_bw, d_body)

        lum = 0.55 + 0.45 * np.clip(d[..., 2], 0, 1)

        # Ground uses the same finite receptor footprint for distance-dependent
        # spatial low-pass filtering. This is still analytic: no extra rays.
        pz = pos[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_g = np.where(d[..., 2] < -1e-6, -pz / d[..., 2], np.inf)
        t_g = np.where((t_g > 0) & (t_g < self.cfg.max_range), t_g, np.inf)
        hit_g = np.isfinite(t_g)
        if hit_g.any():
            p = pos[:, None, :] + np.where(hit_g, t_g, 0.0)[..., None] * d
            k = 2 * np.pi / self.cfg.ground_period
            tex = 0.5 + 0.5 * np.sin(k * p[..., 0]) * np.sin(k * p[..., 1])
            incidence = np.maximum(np.abs(d[..., 2]), 1e-3)
            footprint = (
                np.where(hit_g, t_g, 0.0)
                * self._tan_half_angle
                / np.sqrt(incidence)
            )
            blur = np.clip(
                self.cfg.ground_period / np.maximum(4.0 * footprint, 1e-9), 0, 1
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
            finite = np.isfinite(t)
            hp = pos[:, None, :] + np.where(finite, t, 0.0)[..., None] * d
            rel = hp - c[:, None, :]
            rad = np.linalg.norm(rel, axis=-1)

            # A receptor's angular cell cuts an oblique gate plane as an ellipse.
            # Use the equal-area disk radius (minor radius / sqrt(cos incidence))
            # so annulus coverage can be integrated exactly with two circle-circle
            # overlap evaluations. This costs O(E*C*G), the same asymptotic work as
            # the old center-ray test, instead of multiplying it by 7-13 sub-rays.
            incidence = np.maximum(np.abs(denom), 1e-3)
            footprint = (
                np.where(finite, t, 0.0)
                * self._tan_half_angle
                / np.sqrt(incidence)
            )
            coverage = _annulus_coverage(
                rad,
                gates.r_in[:, gi : gi + 1],
                gates.r_out[:, gi : gi + 1],
                footprint,
            )
            closer = finite & (coverage > 1e-5) & (t < t_best)
            if closer.any():
                e1 = _perp_batch(nrm)
                e2 = np.cross(nrm, e1)
                ang = np.arctan2(
                    np.einsum("ecj,ej->ec", rel, e2),
                    np.einsum("ecj,ej->ec", rel, e1),
                )
                seg = 0.15 + 0.70 * (np.sin(8 * ang) > 0)
                alpha = np.where(closer, coverage, 0.0)
                lum = lum * (1.0 - alpha) + seg * alpha
                t_best = np.where(closer, t, t_best)

        return np.clip(lum, 0.0, 1.0).astype(np.float32)

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


def _annulus_coverage(
    center_distance: np.ndarray,
    inner_radius: np.ndarray,
    outer_radius: np.ndarray,
    footprint_radius: np.ndarray,
) -> np.ndarray:
    """Fraction of a circular receptor footprint covered by an annulus.

    This is the exact planar area fraction under the equal-area disk
    approximation used in :meth:`CompoundEye.render`. Crucially, it is nonzero
    even when the receptor's *center ray* passes through the gate hole or just
    outside the ring, eliminating point-sampling aliasing at long range.
    """
    d, rin, rout, r = np.broadcast_arrays(
        np.asarray(center_distance, dtype=np.float64),
        np.asarray(inner_radius, dtype=np.float64),
        np.asarray(outer_radius, dtype=np.float64),
        np.maximum(np.asarray(footprint_radius, dtype=np.float64), 1e-9),
    )
    coverage = np.zeros(d.shape, dtype=np.float64)

    # Most columns are nowhere near a ring. Resolve the obvious 0/1 cases with
    # comparisons and pay for acos/sqrt only on the thin boundary set.
    none = (d + r <= rin) | (d - r >= rout)
    full = (d - r >= rin) & (d + r <= rout)
    coverage[full] = 1.0
    partial = ~(none | full)
    if partial.any():
        dp, rp = d[partial], r[partial]
        outer = _circle_overlap_area(rout[partial], rp, dp)
        inner = _circle_overlap_area(rin[partial], rp, dp)
        coverage[partial] = (outer - inner) / np.maximum(np.pi * rp * rp, 1e-18)
    return np.clip(coverage, 0.0, 1.0)


def _circle_overlap_area(R: np.ndarray, r: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Vectorised exact overlap area of two circles."""
    R, r, d = np.broadcast_arrays(
        np.asarray(R, dtype=np.float64),
        np.asarray(r, dtype=np.float64),
        np.asarray(d, dtype=np.float64),
    )
    R = np.maximum(R, 0.0)
    r = np.maximum(r, 0.0)
    d = np.maximum(d, 0.0)

    no_overlap = d >= (R + r)
    contained = d <= np.abs(R - r)

    ds = np.maximum(d, 1e-15)
    Rs = np.maximum(R, 1e-15)
    rs = np.maximum(r, 1e-15)
    c1 = np.clip((ds * ds + Rs * Rs - rs * rs) / (2.0 * ds * Rs), -1.0, 1.0)
    c2 = np.clip((ds * ds + rs * rs - Rs * Rs) / (2.0 * ds * rs), -1.0, 1.0)
    heron = np.maximum(
        (-ds + Rs + rs) * (ds + Rs - rs) * (ds - Rs + rs) * (ds + Rs + rs),
        0.0,
    )
    partial = Rs * Rs * np.arccos(c1) + rs * rs * np.arccos(c2) - 0.5 * np.sqrt(heron)
    full_small = np.pi * np.minimum(R, r) ** 2
    return np.where(no_overlap, 0.0, np.where(contained, full_small, partial))


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
