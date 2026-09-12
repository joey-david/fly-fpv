"""A compound eye that renders onto the connectome's own retinotopic grid.

The MaleCNS annotations give every medulla columnar neuron an `assignedOlHex1/2`
coordinate — the ommatidium its column belongs to.  There are ~800 such columns
per eye, which is the real ommatidial count of *Drosophila*.  So we do not
invent an input layer: we take the hex lattice straight out of the connectome,
assign each hex a viewing direction, and ray-cast the world through it.  Neuron
`i` sees exactly the patch of sky its biological counterpart sees.

Two details that matter for flight and that a naive camera would lose:

* **Optic flow needs texture.**  A flying insect regulates altitude and ground
  speed from the rate at which ground texture streams past.  A blank floor is
  invisible to a motion detector, so the ground is rendered with a real pattern.
* **The first synapse splits ON from OFF.**  R1-R6 are histaminergic and
  therefore *inhibit* the lamina monopolar cells, so L1/L2 depolarise to
  contrast decrements.  L1 (with L5, Mi1) feeds the ON pathway to T4; L2 (with
  L3, Tm1/Tm2/Tm9) feeds the OFF pathway to T5.  We drive those two groups with
  rectified positive and negative temporal contrast respectively.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Which retinotopic cell types belong to which half of the ON/OFF split.
ON_TYPES = {"L1", "L5", "Mi1"}
OFF_TYPES = {"L2", "L3", "Tm1", "Tm2", "Tm9"}


@dataclass
class EyeConfig:
    fov_azimuth: tuple[float, float] = (-15.0, 165.0)  # deg, per eye, outward from midline
    fov_elevation: tuple[float, float] = (-60.0, 75.0)
    acceptance_angle: float = 5.0     # deg, Delta-rho: optical blur of one ommatidium
    contrast_tau: float = 0.020       # s, photoreceptor high-pass time constant
    ground_period: float = 0.04       # m, spatial period of the floor texture
    max_range: float = 4.0            # m


class CompoundEye:
    """Ray-casts a hoop course through a hex ommatidial lattice.

    Vectorised over environments. One instance serves both eyes.
    """

    def __init__(self, hex_coords: dict[int, tuple[int, int, str]], n_envs: int,
                 cfg: EyeConfig | None = None):
        self.cfg = cfg or EyeConfig()
        self.n_envs = n_envs

        # Collapse the per-neuron hex map into a per-*column* lattice: many cell
        # types share one ommatidium, and they all see the same thing.
        cols: dict[tuple[int, int, str], list[int]] = {}
        for nid, (h1, h2, side) in hex_coords.items():
            cols.setdefault((h1, h2, side), []).append(nid)
        self.columns = sorted(cols.keys())
        self.column_neurons = [np.array(cols[k], dtype=np.int64) for k in self.columns]
        self.n_columns = len(self.columns)

        self.col_side = np.array([1 if k[2] == "R" else -1 for k in self.columns])
        self.directions = self._hex_to_directions()      # (n_columns, 3) body frame

        # index of each column into a dense per-neuron drive vector
        self._on_rows, self._on_cols = [], []
        self._off_rows, self._off_cols = [], []

        self.prev_luminance = np.zeros((n_envs, self.n_columns), dtype=np.float32)
        self._initialised = np.zeros(n_envs, dtype=bool)

    # ------------------------------------------------------------------
    def _hex_to_directions(self) -> np.ndarray:
        """Map hex lattice coordinates to unit viewing directions in the body frame.

        Axial hex coords -> planar cartesian -> equal-area-ish projection onto
        the eye's viewing patch. Real *Drosophila* optics are non-uniform (the
        frontal 'love spot' is oversampled) but the topology — a hexagonal sheet
        smoothly covering a near-hemisphere — is what the motion circuitry cares
        about, and that is preserved exactly.
        """
        h1 = np.array([k[0] for k in self.columns], dtype=np.float64)
        h2 = np.array([k[1] for k in self.columns], dtype=np.float64)
        # axial -> cartesian on a unit hex lattice
        x = h1 + 0.5 * h2
        y = (np.sqrt(3) / 2) * h2

        dirs = np.zeros((self.n_columns, 3))
        for s in (1, -1):
            m = self.col_side == s
            if not m.any():
                continue
            xs, ys = x[m], y[m]
            u = _unit_range(xs)
            v = _unit_range(ys)
            az0, az1 = np.deg2rad(self.cfg.fov_azimuth)
            el0, el1 = np.deg2rad(self.cfg.fov_elevation)
            az = (az0 + u * (az1 - az0)) * s      # mirrored for the left eye
            el = el0 + v * (el1 - el0)
            dirs[m] = np.stack([
                np.cos(el) * np.cos(az),
                np.cos(el) * np.sin(az),
                np.sin(el),
            ], -1)
        return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    def render(self, pos: np.ndarray, R_bw: np.ndarray, gates, head: np.ndarray) -> np.ndarray:
        """Luminance per ommatidium.

        pos   (E,3) world position of the head
        R_bw  (E,3,3) body -> world rotation
        gates GateArray with .center (G,3), .normal (G,3), .r_in, .r_out
        head  (E,2) head pitch/yaw relative to the body (gaze stabilisation)
        Returns (E, n_columns) in [0, 1].
        """
        E, C = pos.shape[0], self.n_columns
        R_head = _rot_yz(head[:, 0], head[:, 1])                     # (E,3,3)
        d_body = np.einsum("eij,cj->eci", R_head, self.directions)   # (E,C,3)
        d = np.einsum("eij,ecj->eci", R_bw, d_body)                  # world rays

        # sky background: brighter toward the zenith
        lum = 0.55 + 0.45 * np.clip(d[..., 2], 0, 1)

        # --- ground plane z = 0, with texture so motion detectors have signal --
        pz = pos[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_g = np.where(d[..., 2] < -1e-6, -pz / d[..., 2], np.inf)
        t_g = np.where((t_g > 0) & (t_g < self.cfg.max_range), t_g, np.inf)
        hit_g = np.isfinite(t_g)
        if hit_g.any():
            p = pos[:, None, :] + np.where(hit_g, t_g, 0.0)[..., None] * d
            k = 2 * np.pi / self.cfg.ground_period
            tex = 0.5 + 0.5 * np.sin(k * p[..., 0]) * np.sin(k * p[..., 1])
            # fade with distance: an ommatidium's acceptance cone blurs fine
            # texture out, exactly as a real eye's low-pass does
            blur = np.clip(self.cfg.ground_period /
                           (np.deg2rad(self.cfg.acceptance_angle) * np.maximum(t_g, 1e-6) * 4), 0, 1)
            g = 0.25 + 0.45 * (tex * blur + 0.5 * (1 - blur))
            lum = np.where(hit_g, g, lum)
            t_best = np.where(hit_g, t_g, np.inf)
        else:
            t_best = np.full((E, C), np.inf)

        # --- gates as flat annuli: exact, and cheap ---------------------------
        # gates.* are per-environment, (E, G, ...), so every env can race its
        # own course in the same vectorised batch.
        for gi in range(gates.center.shape[1]):
            c = gates.center[:, gi, :]                      # (E,3)
            nrm = gates.normal[:, gi, :]                    # (E,3)
            denom = np.einsum("ecj,ej->ec", d, nrm)
            ok = np.abs(denom) > 1e-6
            num = np.einsum("ej,ej->e", c - pos, nrm)[:, None]
            t = np.where(ok, num / np.where(ok, denom, 1.0), np.inf)
            t = np.where((t > 0) & (t < self.cfg.max_range), t, np.inf)
            hp = pos[:, None, :] + np.where(np.isfinite(t), t, 0.0)[..., None] * d
            rad = np.linalg.norm(hp - c[:, None, :], axis=-1)
            on_ring = (np.isfinite(t) & (rad >= gates.r_in[:, gi : gi + 1])
                       & (rad <= gates.r_out[:, gi : gi + 1]))
            closer = on_ring & (t < t_best)
            if closer.any():
                # alternating dark/light segments make the ring's rotation and
                # range legible to a motion detector, like a real race gate
                rel = hp - c[:, None, :]
                e1 = _perp_batch(nrm)                        # (E,3)
                e2 = np.cross(nrm, e1)
                ang = np.arctan2(np.einsum("ecj,ej->ec", rel, e2),
                                 np.einsum("ecj,ej->ec", rel, e1))
                seg = 0.15 + 0.70 * (np.sin(8 * ang) > 0)
                lum = np.where(closer, seg, lum)
                t_best = np.where(closer, t, t_best)

        return lum.astype(np.float32)

    # ------------------------------------------------------------------
    def photoreceptor_drive(self, luminance: np.ndarray, dt: float, reset_mask=None):
        """Split luminance into the ON and OFF channels the lamina actually carries.

        Returns (on, off), each (E, n_columns) and non-negative. The steady
        (non-transient) component is retained at low gain, because a fly can
        still see a stationary edge.
        """
        if reset_mask is not None and reset_mask.any():
            self.prev_luminance[reset_mask] = luminance[reset_mask]
        a = dt / (self.cfg.contrast_tau + dt)
        delta = (luminance - self.prev_luminance) / max(dt, 1e-6)
        self.prev_luminance += a * (luminance - self.prev_luminance)

        c = np.tanh(delta * self.cfg.contrast_tau)     # bounded temporal contrast
        static = luminance - 0.5
        on = np.maximum(c, 0) + 0.3 * np.maximum(static, 0)
        off = np.maximum(-c, 0) + 0.3 * np.maximum(-static, 0)
        return on.astype(np.float32), off.astype(np.float32)

    def scatter_to_neurons(self, on, off, n_neurons: int, neuron_type: np.ndarray) -> np.ndarray:
        """Place the ON/OFF channels onto their own retinotopic neurons."""
        drive = np.zeros((on.shape[0], n_neurons), dtype=np.float32)
        for ci, rows in enumerate(self.column_neurons):
            if not len(rows):
                continue
            t = neuron_type[rows]
            is_on = np.isin(t, list(ON_TYPES))
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
    """Any unit vector perpendicular to each row of n, chosen stably."""
    a = np.where((np.abs(n[:, 2:3]) < 0.9), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    v = np.cross(n, a)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _rot_yz(pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    z, o = np.zeros_like(cp), np.ones_like(cp)
    Ry = np.stack([np.stack([cp, z, sp], -1), np.stack([z, o, z], -1), np.stack([-sp, z, cp], -1)], -2)
    Rz = np.stack([np.stack([cy, -sy, z], -1), np.stack([sy, cy, z], -1), np.stack([z, z, o], -1)], -2)
    return Rz @ Ry
