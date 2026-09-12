"""3D hoop courses, scaled to an animal 2.5 mm long.

Everything is in SI. A fly cruises at 0.2-1.0 m/s and turns at up to ~2000 deg/s,
so a course with 3 cm hoops spaced 20 cm apart is, to a fly, roughly what a
drone-racing track is to a quadcopter: several body lengths of straight, then a
turn that has to be committed to well before you reach it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GateSpec:
    r_in: float = 0.1  # m, clear opening the fly must pass through
    ring_thickness: float = 0.02
    spacing: tuple[float, float] = (0.46, 0.66)  # m between consecutive gates
    turn: float = 0.9  # rad, max heading change per gate
    climb: float = 0.45  # rad, max pitch change per gate
    altitude: tuple[float, float] = (0.10, 0.75)  # m above the floor


@dataclass
class Course:
    center: np.ndarray  # (G,3) gate centres, world
    normal: np.ndarray  # (G,3) unit normal = direction of travel through the gate
    r_in: np.ndarray  # (G,)
    r_out: np.ndarray  # (G,)
    start_pos: np.ndarray  # (3,)
    start_dir: np.ndarray  # (3,)

    @property
    def n_gates(self) -> int:
        return len(self.center)

    @property
    def length(self) -> float:
        pts = np.vstack([self.start_pos, self.center])
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    def to_dict(self):
        return dict(
            center=self.center.tolist(),
            normal=self.normal.tolist(),
            r_in=self.r_in.tolist(),
            r_out=self.r_out.tolist(),
            start_pos=self.start_pos.tolist(),
            start_dir=self.start_dir.tolist(),
        )


def make_course(
    n_gates: int = 8,
    spec: GateSpec | None = None,
    seed: int | None = None,
    difficulty: float = 0.5,
) -> Course:
    """A random but flyable course: a smooth 3D polyline with a hoop at each vertex.

    `difficulty` in [0,1] scales turn sharpness and shrinks the hoops, so a
    curriculum is just a schedule over this one number.
    """
    spec = spec or GateSpec()
    rng = np.random.default_rng(seed)

    turn = spec.turn * (0.25 + 0.75 * difficulty)
    climb = spec.climb * (0.25 + 0.75 * difficulty)
    r_in = spec.r_in * (1.35 - 0.55 * difficulty)

    pos = np.array([0.0, 0.0, 0.30])
    heading = 0.0  # yaw
    pitch = 0.0

    centers, normals = [], []
    for _ in range(n_gates):
        heading += rng.uniform(-turn, turn)
        pitch = np.clip(pitch + rng.uniform(-climb, climb), -0.55, 0.55)
        d = np.array(
            [
                np.cos(pitch) * np.cos(heading),
                np.cos(pitch) * np.sin(heading),
                np.sin(pitch),
            ]
        )
        step = rng.uniform(*spec.spacing)
        pos = pos + d * step
        pos[2] = np.clip(pos[2], *spec.altitude)
        centers.append(pos.copy())
        normals.append(d)

    centers = np.array(centers)
    # A gate's normal should bisect the incoming and outgoing legs, the way a
    # real race gate is angled into the turn.
    pts = np.vstack([np.array([0.0, 0.0, 0.30]), centers])
    legs = np.diff(pts, axis=0)
    legs /= np.linalg.norm(legs, axis=1, keepdims=True)
    nrm = legs.copy()
    nrm[:-1] = legs[:-1] + legs[1:]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

    return Course(
        center=centers,
        normal=nrm,
        r_in=np.full(n_gates, r_in),
        r_out=np.full(n_gates, r_in + spec.ring_thickness),
        start_pos=np.array([0.0, 0.0, 0.30]),
        start_dir=legs[0],
    )
