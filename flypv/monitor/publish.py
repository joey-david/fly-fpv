"""Pack trainer state into frames the dashboard can draw.

Activations are quantised to uint8 and base64'd. At 28,361 neurons and ~15
frames a second, sending them as JSON numbers would be several megabytes a
second of parsing; as bytes it is ~40 KB a frame, and one byte of precision is
more than a screen pixel can show anyway.
"""
from __future__ import annotations

import base64

import numpy as np
import torch

from .bus import BUS

_TRAIL: list = []
_DISPLAY: np.ndarray | None = None


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a)).decode("ascii")


def _quant(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> str:
    q = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1) * 255.0
    return _b64(q.astype(np.uint8))


def _role_of(circuit) -> np.ndarray:
    """0 intrinsic, 1 retina, 2 other sensory, 3 motor, 4 descending, 5 ascending."""
    n = circuit.n
    role = np.zeros(n, dtype=np.uint8)
    sc = circuit.cx.meta.superclass.to_numpy()
    role[np.isin(sc, ["descending_neuron", "descending_neuron_tbc"])] = 4
    role[np.isin(sc, ["ascending_neuron", "sensory_ascending"])] = 5
    for k, v in circuit.afferent.items():
        role[v] = 1 if k == "retina" else 2
    for v in circuit.efferent.values():
        role[v] = 3
    return role


def _layout(circuit) -> np.ndarray:
    """Anatomical soma positions in micrometres, gaps filled by neuropil centroid."""
    m = circuit.cx.meta
    xyz = np.stack([m.soma_x.to_numpy(), m.soma_y.to_numpy(), m.soma_z.to_numpy()], -1).astype(np.float32)
    missing = ~np.isfinite(xyz).all(-1)
    if missing.any():
        # place unlocalised neurons at the centroid of their own cell type, then
        # their superclass, so they land in anatomically sensible company
        for key in ("type", "superclass"):
            g = m[key].to_numpy()
            for val in np.unique(g[missing]):
                sel = (g == val)
                known = sel & ~missing
                if known.sum() >= 3:
                    c = xyz[known].mean(0)
                    fill = sel & missing
                    xyz[fill] = c + np.random.default_rng(0).normal(0, 8, (fill.sum(), 3))
                    missing = missing & ~sel
        xyz[missing] = np.nanmean(xyz[~missing], axis=0)
    return xyz


def display_index(circuit, budget: int = 14000) -> np.ndarray:
    """Choose which neurons the graph view draws.

    The retina is 14,201 of the 28,361 neurons in the flight circuit, so taking
    "all the afferents plus all the efferents" would fill the whole budget with
    eye and leave no brain, no descending axons and no interneurons on screen.
    Instead the small, structurally distinctive populations are kept whole and
    the two big ones — retina and intrinsic — are sampled.
    """
    global _DISPLAY
    if _DISPLAY is not None:
        return _DISPLAY
    n = circuit.n
    if n <= budget:
        _DISPLAY = np.arange(n)
        return _DISPLAY

    rng = np.random.default_rng(0)
    sc = circuit.cx.meta.superclass.to_numpy()

    keep = set()
    for v in circuit.efferent.values():
        keep |= set(int(i) for i in v)                    # every muscle
    for k, v in circuit.afferent.items():
        if k != "retina":
            keep |= set(int(i) for i in v)                # halteres, antennae, strain
    keep |= set(np.flatnonzero(np.isin(
        sc, ["descending_neuron", "ascending_neuron", "sensory_ascending"])).tolist())

    retina = circuit.afferent.get("retina", np.array([], np.int64))
    share = int(0.30 * budget)
    if len(retina):
        pick = rng.choice(retina, size=min(share, len(retina)), replace=False)
        keep |= set(int(i) for i in pick)

    # the retina has had its share; the remaining budget is for the brain
    spoken_for = np.union1d(np.array(sorted(keep), dtype=np.int64),
                            np.asarray(retina, dtype=np.int64))
    rest = np.setdiff1d(np.arange(n), spoken_for)
    room = max(budget - len(keep), 0)
    if room and len(rest):
        keep |= set(rng.choice(rest, size=min(room, len(rest)), replace=False).tolist())

    _DISPLAY = np.array(sorted(keep), dtype=np.int64)
    return _DISPLAY


def publish_static(circuit, env, policy, cfg):
    idx = display_index(circuit)
    xyz = _layout(circuit)[idx]
    role = _role_of(circuit)[idx]
    meta = circuit.cx.meta.iloc[idx]

    # strongest edges among displayed neurons, so the picture shows real
    # pathways rather than a uniform haze
    W = circuit.cx.W.tocoo()
    pos = {int(g): i for i, g in enumerate(idx)}
    mask = np.array([(r in pos and c in pos) for r, c in zip(W.row, W.col)])
    r, c, w = W.row[mask], W.col[mask], W.data[mask]
    order = np.argsort(-np.abs(w))[:12000]
    edges = np.stack([[pos[int(x)] for x in r[order]],
                      [pos[int(x)] for x in c[order]]], -1).astype(np.int32)
    esign = (w[order] > 0).astype(np.uint8)

    # ommatidial hex layout, both eyes
    cols = sorted({v for v in circuit.hex_coords.values()})
    hx = np.array([h1 + 0.5 * h2 for (h1, h2, _) in cols], dtype=np.float32)
    hy = np.array([(3 ** 0.5 / 2) * h2 for (_, h2, _) in cols], dtype=np.float32)
    hside = np.array([1 if s == "R" else 0 for (_, _, s) in cols], dtype=np.uint8)

    BUS.set_static("graph", dict(
        n=int(len(idx)),
        x=np.round(xyz[:, 0], 1).tolist(),
        y=np.round(xyz[:, 1], 1).tolist(),
        z=np.round(xyz[:, 2], 1).tolist(),
        role=role.tolist(),
        edges=edges.ravel().tolist(),
        edge_sign=esign.tolist(),
        types=[t or sc for t, sc in zip(meta["type"], meta["superclass"])],
    ))
    BUS.set_static("eye", dict(x=hx.tolist(), y=hy.tolist(), side=hside.tolist(),
                               n=len(cols)))
    BUS.set_static("info", dict(
        dataset=circuit.cx.source,
        neurons=int(circuit.n),
        connections=int(circuit.cx.W.nnz),
        synapses=int(circuit.cx.n_synapses),
        displayed=int(len(idx)),
        trainable=int(policy.n_trainable()),
        arch=cfg.arch, scale=cfg.scale,
        n_gates=int(env.cfg.n_gates),
        afferent={k: int(len(v)) for k, v in circuit.afferent.items()},
        efferent={k: int(len(v)) for k, v in circuit.efferent.items()},
        controls=__import__("flypv.wing", fromlist=["WING_CONTROLS"]).WING_CONTROLS,
    ))


def publish_frame(env, policy, circuit, state: torch.Tensor, info: dict):
    """One live frame: neural activity, the eye, and the flight itself."""
    global _TRAIL
    idx = display_index(circuit)

    h = state[:, 0].detach().cpu().numpy() if state.ndim == 2 else None
    if h is not None and h.size:
        a = h[idx]
        hi = float(np.percentile(a, 99.5)) or 1.0
        BUS.publish("act", dict(data=_quant(a, 0.0, max(hi, 1e-3)), hi=hi))

        # per-group mean firing rate: the readout the motor decoder actually uses
        groups = {k: float(h[v].mean()) for k, v in circuit.efferent.items() if len(v)}
        ports = {k: float(h[v].mean()) for k, v in circuit.afferent.items() if len(v)}
        BUS.publish("pools", dict(motor=groups, sensory=ports))

    snap = env.snapshot(0)
    _TRAIL.append(snap["pos"])
    if len(_TRAIL) > 900:
        _TRAIL = _TRAIL[-900:]
    if snap["next_gate"] == 0 and len(_TRAIL) > 2 and np.linalg.norm(
            np.array(_TRAIL[-1]) - np.array(_TRAIL[-2])) > 0.05:
        _TRAIL = _TRAIL[-1:]
    snap["trail"] = [[round(c, 4) for c in p] for p in _TRAIL[::2]]
    BUS.publish("flight", snap)

    if env.n_columns:
        BUS.publish("eye_frame", dict(
            lum=_quant(env.last_lum[0], 0.0, 1.0),
            on=_quant(env.last_on[0], 0.0, 1.0),
            off=_quant(env.last_off[0], 0.0, 1.0),
        ))

    BUS.publish("live", dict(
        speed=float(info["speed"][0]) if "speed" in info else 0.0,
        power_rel=float(info["power_rel"][0]) if "power_rel" in info else 0.0,
        gates_done=int(info["gates_done"][0]) if "gates_done" in info else 0,
        upright=float(info["upright"][0]) if "upright" in info else 0.0,
        spin=float(info["spin"][0]) if "spin" in info else 0.0,
    ))
