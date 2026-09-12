"""Carve the flight-control circuit out of the whole central nervous system.

The extraction is deliberately anatomical rather than a learned sparsification.
It keeps all named flight sensory and motor populations, the neurons lying on
short sensor-to-motor paths, and a small closure of local recurrent circuitry
around those paths. Persistent neural state can therefore exploit feedback and
multi-wingbeat routes instead of being restricted to an unrealistically narrow
feed-forward corridor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .loader import Connectome, load_connectome

# ---------------------------------------------------------------- motor side

POWER_MN = ["DLMn a, b", "DVMn 1a-c", "DVMn 2a, b"]
STEERING_MN = [
    "b1 MN", "b2 MN", "b3 MN",
    "i1 MN", "i2 MN",
    "iii1 MN", "iii2 MN", "iii3 MN", "iii4 MN",
    "hg1 MN", "hg2 MN", "hg3 MN", "hg4 MN",
    "ps1 MN", "ps2 MN",
    "tp1 MN", "tp2 MN", "tpn MN",
]
WING_MISC_MN = ["MNwm35", "MNwm36"]
ESCAPE_MN = ["TTMn", "STTMm"]

MOTOR_GROUPS: dict[str, dict] = {
    "power":     dict(types=POWER_MN, bilateral=True, channel="wingbeat amplitude / frequency drive"),
    "basalar":   dict(types=["b1 MN", "b2 MN", "b3 MN"], bilateral=True, channel="stroke amplitude + downstroke deviation"),
    "axillary":  dict(types=["i1 MN", "i2 MN", "iii1 MN", "iii2 MN", "iii3 MN", "iii4 MN"], bilateral=True, channel="stroke plane / wing rotation"),
    "hinge":     dict(types=["hg1 MN", "hg2 MN", "hg3 MN", "hg4 MN"], bilateral=True, channel="angle of attack"),
    "pterale":   dict(types=["ps1 MN", "ps2 MN", "tp1 MN", "tp2 MN", "tpn MN"], bilateral=True, channel="stroke offset / upstroke deviation"),
    "wing_misc": dict(types=WING_MISC_MN, bilateral=True, channel="hinge tension"),
    "neck":      dict(superclass="cb_motor", bilateral=True, channel="head pitch / yaw (gaze stabilisation)"),
    "abdomen":   dict(neuromeres=[f"A{i}" for i in range(1, 11)], bilateral=False, channel="abdomen pitch (trim)"),
}

# ------------------------------------------------------------- sensory side

RETINOTOPIC_TYPES = ["L1", "L2", "L3", "L5", "Mi1", "Tm1", "Tm2", "Tm9"]

SENSORY_PORTS: dict[str, dict] = {
    "retina": dict(types=RETINOTOPIC_TYPES, retinotopic=True,
                   signal="per-ommatidium luminance and temporal contrast"),
    "haltere": dict(subclass="haltere", signal="body angular velocity (gyro)"),
    "johnston": dict(type_re=r"^JO-", signal="airspeed / wind"),
    "wing_cs": dict(subclass=["wing", "wing bristle", "campaniform sensilla"],
                    signal="wing strain and load"),
    "hairplate": dict(subclass=["hair plate", "neck"], signal="joint angle / head-body angle"),
    "chordotonal": dict(subclass="chordotonal organ", signal="limb proprioception"),
}


@dataclass
class FlightCircuit:
    """A connectome subgraph partitioned into afferent / intrinsic / efferent."""

    cx: Connectome
    afferent: dict[str, np.ndarray]
    efferent: dict[str, np.ndarray]
    intrinsic: np.ndarray
    hex_coords: dict[int, tuple[int, int, str]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.cx.n

    @property
    def afferent_idx(self) -> np.ndarray:
        return np.unique(np.concatenate([v for v in self.afferent.values() if len(v)]))

    @property
    def efferent_idx(self) -> np.ndarray:
        return np.unique(np.concatenate([v for v in self.efferent.values() if len(v)]))

    def side_of(self, rows: np.ndarray) -> np.ndarray:
        s = self.cx.meta.somaSide.to_numpy()[rows]
        return np.where(s == "R", 1, np.where(s == "L", -1, 0))

    def summary(self) -> str:
        lines = [self.cx.summary(), "", "  afferent ports:"]
        for k, v in self.afferent.items():
            lines.append(f"    {k:<12} {len(v):>6,}   {SENSORY_PORTS[k]['signal']}")
        lines.append("  efferent groups:")
        for k, v in self.efferent.items():
            lines.append(f"    {k:<12} {len(v):>6,}   {MOTOR_GROUPS[k]['channel']}")
        lines.append(f"  intrinsic:     {len(self.intrinsic):>6,}")
        return "\n".join(lines)


def _rows_for_motor(cx: Connectome, spec: dict) -> np.ndarray:
    meta = cx.meta
    motor = meta.superclass.isin(["vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent"])
    if "types" in spec:
        m = motor & meta["type"].isin(spec["types"])
    elif "superclass" in spec:
        m = meta.superclass == spec["superclass"]
    elif "neuromeres" in spec:
        m = motor & meta.somaNeuromere.isin(spec["neuromeres"])
    else:
        m = pd.Series(False, index=meta.index)
    return np.flatnonzero(m.to_numpy())


def _rows_for_sensory(cx: Connectome, spec: dict) -> np.ndarray:
    meta = cx.meta
    if "types" in spec:
        m = meta["type"].isin(spec["types"])
    elif "type_re" in spec:
        m = meta["type"].str.contains(spec["type_re"], regex=True)
    elif "subclass" in spec:
        v = spec["subclass"]
        m = meta.subclass.isin(v if isinstance(v, list) else [v])
    else:
        m = pd.Series(False, index=meta.index)
    return np.flatnonzero(m.to_numpy())


def _binary_adjacency(W: sp.csr_matrix, reverse: bool = False) -> sp.csr_matrix:
    A = (W.T if reverse else W).tocsr(copy=True)
    A.data = np.ones(A.nnz, dtype=np.float32)
    return A


def _khop(W: sp.csr_matrix, seeds: np.ndarray, hops: int, reverse: bool) -> np.ndarray:
    """Reachable set within ``hops`` synapses. W[post, pre]."""
    A = _binary_adjacency(W, reverse)
    x = np.zeros(W.shape[0], dtype=bool)
    x[seeds] = True
    reached = x.copy()
    for _ in range(hops):
        x = (A @ x.astype(np.float32)) > 0
        new = x & ~reached
        if not new.any():
            break
        reached |= new
    return np.flatnonzero(reached)


def _recurrent_closure(W: sp.csr_matrix, rows: np.ndarray, hops: int) -> np.ndarray:
    """Add nearby neurons that participate in feedback around the corridor.

    A neuron is admitted when it both receives from and sends back into the
    current set. This preserves local recurrent side-loops without exploding to
    every one-hop neighbour of a high-degree visual neuron. Repeating the rule
    grows nested feedback loops while keeping the operation sparse.
    """
    if hops <= 0 or not len(rows):
        return np.asarray(rows, dtype=np.int64)
    out_adj = _binary_adjacency(W, reverse=False)
    in_adj = _binary_adjacency(W, reverse=True)
    active = np.zeros(W.shape[0], dtype=bool)
    active[rows] = True
    for _ in range(hops):
        x = active.astype(np.float32)
        receives_from_active = (out_adj @ x) > 0
        sends_to_active = (in_adj @ x) > 0
        add = receives_from_active & sends_to_active & ~active
        if not add.any():
            break
        active |= add
    return np.flatnonzero(active)


def _prune_to_budget(cx, rows, afferent, eff_all, budget, verbose):
    """Shrink a candidate circuit while protecting all irreplaceable ports."""
    rng = np.random.default_rng(0)
    keep = set(int(i) for i in eff_all)

    small = np.concatenate([v for k, v in afferent.items() if k != "retina" and len(v)]) \
        if any(k != "retina" and len(v) for k, v in afferent.items()) else np.array([], np.int64)
    keep |= set(int(i) for i in small)

    retina = afferent.get("retina", np.array([], np.int64))
    if len(retina):
        share = int(0.30 * budget)
        hex1 = cx.meta.assignedOlHex1.to_numpy()
        hex2 = cx.meta.assignedOlHex2.to_numpy()
        side = cx.meta.somaSide.to_numpy()
        by_col: dict = {}
        for i in retina:
            by_col.setdefault((hex1[i], hex2[i], side[i]), []).append(int(i))
        cols = list(by_col)
        rng.shuffle(cols)
        used = 0
        for c in cols:
            if used + len(by_col[c]) > share:
                continue
            keep |= set(by_col[c])
            used += len(by_col[c])
        if verbose:
            print(f"[circuit] retina subsampled to {used:,} neurons "
                  f"across {sum(1 for c in cols if by_col[c][0] in keep):,} ommatidial columns")

    if len(keep) < budget:
        sub = cx.W[rows][:, rows]
        thru = (np.asarray(np.abs(sub).sum(axis=0)).ravel()
                + np.asarray(np.abs(sub).sum(axis=1)).ravel())
        for j in np.argsort(-thru):
            if len(keep) >= budget:
                break
            keep.add(int(rows[j]))

    out = np.array(sorted(keep), dtype=np.int64)
    if verbose:
        print(f"[circuit] pruned to {len(out):,} neurons within a budget of {budget:,}")
    return out


def build_flight_circuit(
    cx: Connectome | None = None,
    scale: str = "flight",
    hops: int = 3,
    recurrent_closure_hops: int = 1,
    max_neurons: int | None = None,
    verbose: bool = True,
) -> FlightCircuit:
    """Extract the sensorimotor circuit for flight.

    ``flight`` keeps neurons that can lie on a path no more than ``hops`` edges
    from a flight sensor and no more than ``hops`` edges from a flight motor,
    then admits ``recurrent_closure_hops`` layers of locally bidirectional
    feedback. The defaults are intentionally broader than the original 2+2-hop
    corridor because persistent state lets biologically longer routes matter.

    ``core`` applies the same extraction and then prunes to a small throughput
    core. ``full`` keeps the entire traced CNS.
    """
    if hops < 1:
        raise ValueError("hops must be >= 1")
    if recurrent_closure_hops < 0:
        raise ValueError("recurrent_closure_hops must be >= 0")
    if cx is None:
        cx = load_connectome(verbose=verbose)

    afferent = {k: _rows_for_sensory(cx, s) for k, s in SENSORY_PORTS.items()}
    efferent = {k: _rows_for_motor(cx, s) for k, s in MOTOR_GROUPS.items()}

    aff_all = np.unique(np.concatenate([v for v in afferent.values() if len(v)]))
    eff_all = np.unique(np.concatenate([v for v in efferent.values() if len(v)]))
    if verbose:
        print(f"[circuit] seeds: {len(aff_all):,} sensory, {len(eff_all):,} motor")

    if scale == "full":
        rows = np.arange(cx.n)
    else:
        fwd = _khop(cx.W, aff_all, hops, reverse=False)
        bwd = _khop(cx.W, eff_all, hops, reverse=True)
        between = np.intersect1d(fwd, bwd)
        corridor = np.union1d(np.union1d(aff_all, eff_all), between)
        rows = _recurrent_closure(cx.W, corridor, recurrent_closure_hops)
        if verbose:
            added = len(rows) - len(corridor)
            print(f"[circuit] {len(fwd):,} within {hops} hops of a sensor, "
                  f"{len(bwd):,} within {hops} hops of a muscle, "
                  f"{len(between):,} on a path between")
            print(f"[circuit] recurrent closure +{added:,} neurons "
                  f"({recurrent_closure_hops} hop{'s' if recurrent_closure_hops != 1 else ''})")

    if scale == "core" or (max_neurons and len(rows) > max_neurons):
        budget = max_neurons or 8000
        rows = _prune_to_budget(cx, rows, afferent, eff_all, budget, verbose)

    sub = cx.subgraph(rows)
    remap = {int(g): i for i, g in enumerate(rows)}
    afferent = {k: np.array([remap[int(i)] for i in v if int(i) in remap], dtype=np.int64)
                for k, v in afferent.items()}
    efferent = {k: np.array([remap[int(i)] for i in v if int(i) in remap], dtype=np.int64)
                for k, v in efferent.items()}

    used = np.unique(np.concatenate(
        [v for v in list(afferent.values()) + list(efferent.values()) if len(v)]))
    intrinsic = np.setdiff1d(np.arange(sub.n), used)

    hexes = {}
    m = sub.meta
    for i in afferent.get("retina", []):
        h1, h2 = m.assignedOlHex1.iat[int(i)], m.assignedOlHex2.iat[int(i)]
        if not (np.isnan(h1) or np.isnan(h2)):
            hexes[int(i)] = (int(h1), int(h2), m.somaSide.iat[int(i)] or "?")

    fc = FlightCircuit(cx=sub, afferent=afferent, efferent=efferent,
                       intrinsic=intrinsic, hex_coords=hexes)
    if verbose:
        print("[circuit]\n" + fc.summary())
    return fc
