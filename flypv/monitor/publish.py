"""Pack trainer state into compact frames for the live dashboard."""
from __future__ import annotations

import base64

import numpy as np
import torch

from .bus import BUS

_TRAIL: list = []
_LAST_POS: np.ndarray | None = None
_DISPLAY: np.ndarray | None = None


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a)).decode("ascii")


def _quant(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> str:
    q = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1) * 255.0
    return _b64(q.astype(np.uint8))


def _role_of(circuit) -> np.ndarray:
    role = np.zeros(circuit.n, dtype=np.uint8)
    sc = circuit.cx.meta.superclass.to_numpy()
    role[np.isin(sc, ["descending_neuron", "descending_neuron_tbc"])] = 4
    role[np.isin(sc, ["ascending_neuron", "sensory_ascending"])] = 5
    for name, rows in circuit.afferent.items():
        role[rows] = 1 if name == "retina" else 2
    for rows in circuit.efferent.values():
        role[rows] = 3
    return role


def _layout(circuit) -> np.ndarray:
    meta = circuit.cx.meta
    xyz = np.stack(
        [meta.soma_x.to_numpy(), meta.soma_y.to_numpy(), meta.soma_z.to_numpy()], -1
    ).astype(np.float32)
    missing = ~np.isfinite(xyz).all(-1)
    if missing.any():
        rng = np.random.default_rng(0)
        for key in ("type", "superclass"):
            group = meta[key].to_numpy()
            for value in np.unique(group[missing]):
                selected = group == value
                known = selected & ~missing
                if known.sum() >= 3:
                    fill = selected & missing
                    xyz[fill] = xyz[known].mean(0) + rng.normal(0, 8, (fill.sum(), 3))
                    missing &= ~fill
        xyz[missing] = np.nanmean(xyz[~missing], axis=0)
    return xyz


def display_index(circuit, budget: int = 14000) -> np.ndarray:
    global _DISPLAY
    if _DISPLAY is not None:
        return _DISPLAY
    if circuit.n <= budget:
        _DISPLAY = np.arange(circuit.n)
        return _DISPLAY

    rng = np.random.default_rng(0)
    sc = circuit.cx.meta.superclass.to_numpy()
    keep: set[int] = set()
    for rows in circuit.efferent.values():
        keep.update(int(i) for i in rows)
    for name, rows in circuit.afferent.items():
        if name != "retina":
            keep.update(int(i) for i in rows)
    keep.update(np.flatnonzero(
        np.isin(sc, ["descending_neuron", "ascending_neuron", "sensory_ascending"])
    ).tolist())

    retina = circuit.afferent.get("retina", np.array([], np.int64))
    if len(retina):
        pick = rng.choice(retina, size=min(int(0.30 * budget), len(retina)), replace=False)
        keep.update(int(i) for i in pick)

    spoken_for = np.union1d(np.array(sorted(keep), dtype=np.int64), np.asarray(retina, dtype=np.int64))
    rest = np.setdiff1d(np.arange(circuit.n), spoken_for)
    room = max(budget - len(keep), 0)
    if room and len(rest):
        keep.update(rng.choice(rest, size=min(room, len(rest)), replace=False).tolist())

    _DISPLAY = np.array(sorted(keep), dtype=np.int64)
    return _DISPLAY


def publish_static(circuit, env, policy, cfg):
    idx = display_index(circuit)
    xyz = _layout(circuit)[idx]
    role = _role_of(circuit)[idx]
    meta = circuit.cx.meta.iloc[idx]

    W = circuit.cx.W.tocoo()
    pos = {int(g): i for i, g in enumerate(idx)}
    mask = np.array([(r in pos and c in pos) for r, c in zip(W.row, W.col)])
    rows, cols, weights = W.row[mask], W.col[mask], W.data[mask]
    order = np.argsort(-np.abs(weights))[:12000]
    edges = np.stack(
        [[pos[int(x)] for x in rows[order]], [pos[int(x)] for x in cols[order]]], -1
    ).astype(np.int32)

    columns = sorted(set(circuit.hex_coords.values()))
    hx = np.array([h1 + 0.5 * h2 for h1, h2, _ in columns], dtype=np.float32)
    hy = np.array([(3**0.5 / 2) * h2 for _, h2, _ in columns], dtype=np.float32)
    hside = np.array([1 if side == "R" else 0 for _, _, side in columns], dtype=np.uint8)

    BUS.set_static("graph", dict(
        n=int(len(idx)), x=np.round(xyz[:, 0], 1).tolist(),
        y=np.round(xyz[:, 1], 1).tolist(), z=np.round(xyz[:, 2], 1).tolist(),
        role=role.tolist(), edges=edges.ravel().tolist(),
        edge_sign=(weights[order] > 0).astype(np.uint8).tolist(),
        types=[t or sc for t, sc in zip(meta["type"], meta["superclass"])],
    ))
    BUS.set_static("eye", dict(x=hx.tolist(), y=hy.tolist(), side=hside.tolist(), n=len(columns)))

    m = env.m
    eye_cfg = env.eye.cfg if env.eye is not None else None
    eye_fov = None
    if eye_cfg is not None:
        inward, rear = eye_cfg.fov_azimuth
        eye_fov = dict(
            azimuth_deg=[float(inward), float(rear)],
            elevation_deg=[float(eye_cfg.fov_elevation[0]), float(eye_cfg.fov_elevation[1])],
            overlap_deg=float(max(0.0, -2.0 * inward)),
            rear_blind_deg=float(max(0.0, 360.0 - 2.0 * rear)),
            acceptance_deg=float(eye_cfg.acceptance_angle),
        )
    BUS.set_static("info", dict(
        dataset=circuit.cx.source,
        neurons=int(circuit.n), connections=int(circuit.cx.W.nnz),
        synapses=int(circuit.cx.n_synapses), displayed=int(len(idx)),
        trainable=int(policy.n_trainable()), arch=cfg.arch, scale=cfg.scale,
        n_gates=int(env.cfg.n_gates), policy_mode="stateless",
        propagation_iters=int(cfg.n_iters), control_hz=float(env.cfg.control_hz),
        eye_fov=eye_fov,
        morphology=dict(
            wing_length=float(m.wing_length), mean_chord=float(m.mean_chord),
            hinge_offset=m.hinge_offset.tolist(), stroke_plane_angle=float(m.stroke_plane_angle),
            body_length=2.4e-3, body_radius=0.55e-3,
        ),
        afferent={k: int(len(v)) for k, v in circuit.afferent.items()},
        efferent={k: int(len(v)) for k, v in circuit.efferent.items()},
    ))


def _episode_reset(info: dict, pos: np.ndarray) -> bool:
    global _LAST_POS
    reset = False
    for key in ("crashed", "finished"):
        value = info.get(key)
        if value is not None and len(value) and bool(value[0]):
            reset = True
    if _LAST_POS is not None and np.linalg.norm(pos - _LAST_POS) > 0.12:
        reset = True
    _LAST_POS = pos.copy()
    return reset


def publish_frame(env, policy, circuit, state: torch.Tensor, info: dict):
    global _TRAIL
    idx = display_index(circuit)

    if state.ndim == 2 and state.numel():
        h = state[:, 0].detach().cpu().numpy()
        activity = h[idx]
        hi = float(np.percentile(activity, 99.5)) or 1.0
        BUS.publish_many(
            act=dict(data=_quant(activity, 0.0, max(hi, 1e-3)), hi=hi),
            pools=dict(
                motor={k: float(h[v].mean()) for k, v in circuit.efferent.items() if len(v)},
                sensory={k: float(h[v].mean()) for k, v in circuit.afferent.items() if len(v)},
            ),
        )

    snap = env.snapshot(0)
    snap.pop("luminance", None)
    ws = env.wpg.kinematics()
    snap["wing"].update(
        stroke_tilt=float(ws.stroke_tilt[0]),
        head=ws.head[0].tolist(), abdomen=float(ws.abdomen[0]),
    )

    pos = np.asarray(snap["pos"], dtype=np.float64)
    reset = _episode_reset(info, pos)
    if reset:
        _TRAIL = []
    _TRAIL.append(snap["pos"])
    if len(_TRAIL) > 360:
        _TRAIL = _TRAIL[-360:]
    snap["trail"] = [[round(c, 4) for c in p] for p in _TRAIL[::2]]
    snap["reset"] = reset

    channels = {"flight": snap}
    if env.n_columns:
        channels["eye_frame"] = dict(lum=_quant(env.last_lum[0]))
    channels["live"] = dict(
        speed=float(info["speed"][0]) if "speed" in info else 0.0,
        power_rel=float(info["power_rel"][0]) if "power_rel" in info else 0.0,
        gates_done=int(info["gates_done"][0]) if "gates_done" in info else 0,
        upright=float(info["upright"][0]) if "upright" in info else 0.0,
        spin=float(info["spin"][0]) if "spin" in info else 0.0,
    )
    BUS.publish_many(**channels)
