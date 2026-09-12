"""Measured Drosophila melanogaster body and wing parameters.

Sources are the standard flight-biomechanics literature (Fry, Sayaman &
Dickinson 2003/2005; Sane & Dickinson 1999/2002; Muijres et al. 2014).  These
are real numbers, not tuned-for-RL numbers — a fly that is too light or whose
wings are too large is a different control problem, and we want the one
evolution actually solved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlyMorphology:
    mass: float                 # kg
    wing_length: float          # m, hinge to tip
    mean_chord: float           # m
    wing_area: float            # m^2, one wing
    r2_hat: float               # non-dimensional 2nd moment of wing area
    hinge_offset: np.ndarray    # m, wing hinge in body frame (right wing; left mirrors y)
    stroke_plane_angle: float   # rad, stroke plane tilt from body-normal at hover
    inertia: np.ndarray         # kg m^2, diagonal (roll, pitch, yaw) in body frame
    wing_mass: float            # kg, one wing (for added-mass / inertial torque)
    body_drag_area: float       # m^2, C_d * frontal area of the body
    f_nominal: float            # Hz, hovering wingbeat frequency
    amplitude_nominal: float    # rad, peak-to-peak stroke amplitude at hover
    air_density: float = 1.2    # kg/m^3
    gravity: float = 9.81       # m/s^2

    @property
    def weight(self) -> float:
        return self.mass * self.gravity


# Body frame convention: +x forward (head), +y right, +z up.
DROSOPHILA = FlyMorphology(
    mass=0.96e-6,
    wing_length=2.5e-3,
    mean_chord=0.78e-3,
    wing_area=1.95e-6,
    r2_hat=0.58,
    hinge_offset=np.array([0.0, 0.22e-3, 0.35e-3]),
    stroke_plane_angle=np.deg2rad(-15.0),
    # ellipsoid approximation, semi-axes 1.2 mm x 0.55 mm x 0.55 mm:
    # Ixx = (2/5)m b^2, Iyy = (1/5)m(a^2+b^2). Order 1e-13 kg m^2, matching
    # the published Drosophila values (Cheng & Deng 2011; Ristroph et al. 2013).
    inertia=np.array([1.16e-13, 3.34e-13, 3.34e-13]),
    wing_mass=2.4e-9,
    body_drag_area=0.6 * np.pi * (0.55e-3) ** 2,
    f_nominal=218.0,
    amplitude_nominal=np.deg2rad(140.0),
)
