"""Extensions to the legacy monitor telemetry for the persistent-state dashboard."""
from __future__ import annotations

import torch

from .bus import BUS
from . import publish as legacy


def publish_static(circuit, env, policy, cfg):
    """Publish the existing graph/eye payload, then add dynamics metadata."""
    legacy.publish_static(circuit, env, policy, cfg)
    static = BUS.static()
    info = dict(static.get("info", {}))
    m = env.m
    info["recurrent"] = bool(getattr(cfg, "recurrent", False))
    info["tbptt_steps"] = int(getattr(cfg, "tbptt_steps", 0))
    info["control_hz"] = float(env.cfg.control_hz)
    info["state_carry"] = float(getattr(cfg, "state_carry", 0.0))
    info["morphology"] = dict(
        wing_length=float(m.wing_length),
        mean_chord=float(m.mean_chord),
        hinge_offset=m.hinge_offset.tolist(),
        stroke_plane_angle=float(m.stroke_plane_angle),
        body_length=2.4e-3,
        body_radius=0.55e-3,
    )
    BUS.set_static("info", info)


def publish_frame(env, policy, circuit, state: torch.Tensor, info: dict):
    """Publish the legacy frame and augment it with exact instantaneous kinematics."""
    legacy.publish_frame(env, policy, circuit, state, info)
    latest = BUS.snapshot().get("flight")
    if not latest:
        return

    snap = dict(latest)
    wing = dict(snap.get("wing", {}))
    ws = env.wpg.kinematics()
    wing.update(
        stroke_tilt=float(ws.stroke_tilt[0]),
        head=ws.head[0].tolist(),
        abdomen=float(ws.abdomen[0]),
    )
    snap["wing"] = wing
    BUS.publish("flight", snap)
