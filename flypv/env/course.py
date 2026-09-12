"""Procedural 3D hoop courses in SI units.

The default geometry is deliberately generous enough for early RL: 10 cm clear
radius, 2 cm ring thickness, and 46–66 cm spacing before the difficulty scalar
shrinks gates and increases turns/climbs. These numbers are task parameters,
not claims about a uniquely biologically natural racing course.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GateSpec:
    r_in: float = 0.1
    ring_thickness: float = 0.02
    spacing: tuple[float, float] = (0.46, 0.66)
    turn: float = 0.9
    climb: float = 0.45
    altitude: tuple[float, float] = (0.10, 0.75)


@dataclass
class Course:
    center: np.ndarray
    normal: np.ndarray
    r_in: np.ndarray
    r_out: np.ndarray
    start_pos: np.ndarray
    start_dir: np.ndarray

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
    """Generate a smooth random course.

    ``difficulty`` in [0, 1] scales turn/climb sharpness and gate radius.
    """
    spec = spec or GateSpec()
    rng = np.random.default_rng(seed)

    turn = spec.turn * (0.25 + 0.75 * difficulty)
    climb = spec.climb * (0.25 + 0.75 * difficulty)
    r_in = spec.r_in * (1.35 - 0.55 * difficulty)

    pos = np.array([0.0, 0.0, 0.30])
    heading = 0.0
    pitch = 0.0

    centers = []
    for _ in range(n_gates):
        heading += rng.uniform(-turn, turn)
        pitch = np.clip(pitch + rng.uniform(-climb, climb), -0.55, 0.55)
        direction = np.array(
            [
                np.cos(pitch) * np.cos(heading),
                np.cos(pitch) * np.sin(heading),
                np.sin(pitch),
            ]
        )
        pos = pos + direction * rng.uniform(*spec.spacing)
        pos[2] = np.clip(pos[2], *spec.altitude)
        centers.append(pos.copy())

    centers = np.array(centers)
    start = np.array([0.0, 0.0, 0.30])
    pts = np.vstack([start, centers])
    legs = np.diff(pts, axis=0)
    legs /= np.linalg.norm(legs, axis=1, keepdims=True)

    # Angle gates into turns by bisecting adjacent course legs.
    normals = legs.copy()
    normals[:-1] = legs[:-1] + legs[1:]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return Course(
        center=centers,
        normal=normals,
        r_in=np.full(n_gates, r_in),
        r_out=np.full(n_gates, r_in + spec.ring_thickness),
        start_pos=start,
        start_dir=legs[0],
    )
