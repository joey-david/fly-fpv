"""Wing pattern generator: the oscillator the connectome steers.

Drosophila flight muscle is *asynchronous*.  The power muscles (DLM/DVM) are
stretch-activated and fire roughly once every few wingbeats; the thorax is a
mechanically resonant box that rings at ~218 Hz on its own.  No neuron in the
fly commands "wing up now".  What the nervous system does is (a) set the
oscillator's amplitude and frequency through the power muscles, and (b) retune
the wing hinge stroke-by-stroke through ~17 tiny steering muscles per side.

So the WPG here is not a crutch to make RL easier — it is the correct model of
the plant.  The policy outputs the same 15 knobs a real fly has, and the
oscillator supplies the beat.

Control layout (all in [-1, 1], symmetric residuals about hover trim):
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .morphology import DROSOPHILA, FlyMorphology

WING_CONTROLS: list[str] = [
    "freq",              # 0   power muscles, bilateral: wingbeat frequency
    "amp_R", "amp_L",    # 1,2 power + b1: stroke amplitude per side -> yaw/roll
    "offset_R", "offset_L",   # 3,4 stroke mean angle -> pitch
    "dev_R", "dev_L",    # 5,6 out-of-plane deviation (figure-eight shape)
    "aoa_down_R", "aoa_down_L",  # 7,8   downstroke angle of attack (hg, iii)
    "aoa_up_R", "aoa_up_L",      # 9,10  upstroke angle of attack -> rotation timing
    "stroke_tilt",       # 11  stroke-plane tilt -> forward thrust
    "head_pitch", "head_yaw",    # 12,13 gaze stabilisation (moves the eyes)
    "abdomen_pitch",     # 14  abdomen as a trim flap / inertial appendage
]
N_WING_CONTROLS = len(WING_CONTROLS)

# Physiological travel of each control away from hover trim.  Sustained wild-type
# D. melanogaster flight is typically ~170--220 Hz.  The old +-22% frequency
# range let a 218 Hz fly sit at 266 Hz indefinitely and, combined with maximum
# stroke amplitude, the quasi-steady model could produce ~2.7 body weights of
# beat-averaged force.  Restricting frequency to +-8% gives ~201--235 Hz and a
# maximum symmetric force envelope of about 2.1 body weights, matching measured
# short-burst free-flight capacity while retaining a little headroom above hover.
_SCALE = np.array([
    0.08,                    # freq: +-8% of 218 Hz -> ~201--235 Hz
    0.35, 0.35,              # amplitude: +-35% of 140 deg
    0.40, 0.40,              # stroke offset: +-0.40 rad
    0.30, 0.30,              # deviation amplitude: +-0.30 rad
    0.60, 0.60,              # downstroke AoA: +-0.60 rad
    0.60, 0.60,              # upstroke AoA: +-0.60 rad
    0.50,                    # stroke plane tilt: +-0.50 rad
    0.50, 0.50,              # head: +-0.50 rad
    0.60,                    # abdomen: +-0.60 rad
])

ALPHA_MID = np.deg2rad(45.0)   # hover angle of attack at mid-stroke
_FLIP_SHARPNESS = 4.0          # how abruptly the wing supinates at reversal


#: How far the centre of mass travels per radian of abdomen deflection, in metres.
#: The abdomen is roughly a third of body mass on a ~1 mm arm.
ABDOMEN_COM_ARM = 0.30e-3


@dataclass
class WingState:
    phase: np.ndarray      # (E,) radians
    freq: np.ndarray       # (E,) Hz
    phi: np.ndarray        # (E,2)
    theta: np.ndarray      # (E,2)
    alpha: np.ndarray      # (E,2)
    dphi: np.ndarray       # (E,2)
    dtheta: np.ndarray     # (E,2)
    dalpha: np.ndarray     # (E,2)
    stroke_tilt: np.ndarray  # (E,)
    head: np.ndarray       # (E,2) pitch, yaw
    abdomen: np.ndarray    # (E,)


class WingPatternGenerator:
    """Maps a held control vector + a phase to instantaneous wing kinematics."""

    def __init__(self, n_envs: int, morph: FlyMorphology = DROSOPHILA, trim: bool = True):
        self.n = n_envs
        self.m = morph
        # Zero action must mean hover, not "textbook kinematics that tumble".
        from .trim import solve_trim
        self.trim_offset, self.trim_amp, self.trim_tilt = (
            solve_trim(morph) if trim else (0.0, 1.0, 0.0))
        self.phase = np.zeros(n_envs, dtype=np.float64)
        self.controls = np.zeros((n_envs, N_WING_CONTROLS), dtype=np.float64)

    def reset(self, mask: np.ndarray | None = None, rng: np.random.Generator | None = None):
        mask = np.ones(self.n, dtype=bool) if mask is None else mask
        rng = rng or np.random.default_rng()
        self.phase[mask] = rng.uniform(0, 2 * np.pi, mask.sum())
        self.controls[mask] = 0.0

    def set_controls(self, u: np.ndarray):
        """u: (E, 15) in [-1, 1]."""
        self.controls = np.clip(u, -1.0, 1.0) * _SCALE[None, :]

    def advance(self, dt: float) -> WingState:
        """Step the oscillator by dt and return instantaneous kinematics."""
        c = self.controls
        f = self.m.f_nominal * (1.0 + c[:, 0])
        self.phase = (self.phase + 2 * np.pi * f * dt) % (2 * np.pi)
        return self.kinematics(f)

    def kinematics(self, f: np.ndarray | None = None) -> WingState:
        c = self.controls
        if f is None:
            f = self.m.f_nominal * (1.0 + c[:, 0])
        p = self.phase
        w = 2 * np.pi * f                       # rad/s

        amp = (self.m.amplitude_nominal * self.trim_amp
               * (1.0 + np.stack([c[:, 1], c[:, 2]], -1)))              # (E,2)
        off = self.trim_offset + np.stack([c[:, 3], c[:, 4]], -1)
        dev = np.stack([c[:, 5], c[:, 6]], -1)
        aoa_d = np.stack([c[:, 7], c[:, 8]], -1)
        aoa_u = np.stack([c[:, 9], c[:, 10]], -1)

        sp, cp = np.sin(p)[:, None], np.cos(p)[:, None]
        s2p, c2p = np.sin(2 * p)[:, None], np.cos(2 * p)[:, None]
        w_ = w[:, None]

        # sweep
        phi = off + 0.5 * amp * sp
        dphi = 0.5 * amp * cp * w_

        # out-of-plane deviation: second harmonic gives the classic figure-eight
        theta = dev * s2p
        dtheta = 2.0 * dev * c2p * w_

        # supination/pronation: a smoothed square wave that flips at reversal
        s = np.tanh(_FLIP_SHARPNESS * cp)
        ds = (1 - s**2) * _FLIP_SHARPNESS * (-sp) * w_
        down = 0.5 * (1 + s)
        alpha = (ALPHA_MID + aoa_d) * down - (ALPHA_MID + aoa_u) * (1 - down)
        dalpha = 0.5 * ds * ((ALPHA_MID + aoa_d) + (ALPHA_MID + aoa_u))

        return WingState(
            phase=p.copy(), freq=f, phi=phi, theta=theta, alpha=alpha,
            dphi=dphi, dtheta=dtheta, dalpha=dalpha,
            stroke_tilt=self.trim_tilt + c[:, 11],
            head=np.stack([c[:, 12], c[:, 13]], -1),
            abdomen=c[:, 14],
        )
