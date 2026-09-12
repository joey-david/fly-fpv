"""Carve the flight control circuit out of the whole central nervous system.

A fly does not fly with its whole brain, and we should not pretend otherwise.
This module names the actual anatomical entry and exit points of flight control
and then extracts the tissue between them.

**Exits (efferent).**  Insect flight muscle splits into two functional classes
and the connectome names them explicitly:

  * *Power muscles* — DLMn (dorsal longitudinal) and DVMn (dorsoventral).  These
    are asynchronous stretch-activated muscles.  They do not fire once per
    wingbeat; they set the *amplitude and frequency* of an oscillation the
    thorax sustains mechanically.  That is why our wing-pattern generator is a
    resonant oscillator the policy modulates, not something it drives spike by
    spike.
  * *Steering muscles* — b1/b2/b3 (basalar), i1/i2 and iii1/iii3 (axillary),
    hg1-hg4 (hinge), ps1/ps2, tp1/tp2/tpn.  Each is a single muscle with a
    single motor neuron, firing phase-locked to the beat, and together they
    retune the wing hinge stroke by stroke.  These are the steering knobs.

**Entrances (afferent).**

  * *Halteres* — the hindwings evolved into gyroscopes.  They give the fly
    Coriolis-derived body angular velocity at wingbeat latency, which is exactly
    the signal a drone's rate gyro provides.
  * *Optic lobe columns* — the medulla columnar cells (L1/L2/L3/L5, Mi1, Tm1,
    Tm2, Tm9) carry `assignedOlHex1/2`, the ommatidial coordinate of the column
    they belong to.  ~800 columns per eye, matching the real eye.  That lets us
    project a rendered retina onto the *correct* neurons instead of an arbitrary
    input layer.
  * *Johnston's organ* — antennal mechanoreceptors, the fly's airspeed sensor.
  * *Wing campaniform sensilla and hair plates* — strain and load feedback.
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

#: Motor groups -> the control channel each one owns.  Order matters: this is
#: the layout of the efferent readout vector.
MOTOR_GROUPS: dict[str, dict] = {
    "power":     dict(types=POWER_MN,      bilateral=True,  channel="wingbeat amplitude / frequency drive"),
    "basalar":   dict(types=["b1 MN", "b2 MN", "b3 MN"], bilateral=True, channel="stroke amplitude + downstroke deviation"),
    "axillary":  dict(types=["i1 MN", "i2 MN", "iii1 MN", "iii2 MN", "iii3 MN", "iii4 MN"], bilateral=True, channel="stroke plane / wing rotation"),
    "hinge":     dict(types=["hg1 MN", "hg2 MN", "hg3 MN", "hg4 MN"], bilateral=True, channel="angle of attack"),
    "pterale":   dict(types=["ps1 MN", "ps2 MN", "tp1 MN", "tp2 MN", "tpn MN"], bilateral=True, channel="stroke offset / upstroke deviation"),
    "wing_misc": dict(types=WING_MISC_MN, bilateral=True,  channel="hinge tension"),
    "neck":      dict(superclass="cb_motor", bilateral=True, channel="head pitch / yaw (gaze stabilisation)"),
    "abdomen":   dict(neuromeres=[f"A{i}" for i in range(1, 11)], bilateral=False, channel="abdomen pitch (trim)"),
}

# ------------------------------------------------------------- sensory side

#: Medulla/lamina columnar cell types that carry a retinotopic hex address.
#: L1/L2 are the ON/OFF split at the first synapse; Mi1/Tm1/Tm2/Tm9 are the
#: inputs to the T4/T5 elementary motion detectors.
RETINOTOPIC_TYPES = ["L1", "L2", "L3", "L5", "Mi1", "Tm1", "Tm2", "Tm9"]

SENSORY_PORTS: dict[str, dict] = {
    "retina":    dict(types=RETINOTOPIC_TYPES, retinotopic=True,
                      signal="per-ommatidium luminance and temporal contrast"),
    "haltere":   dict(subclass="haltere",            signal="body angular velocity (gyro)"),
    "johnston":  dict(type_re=r"^JO-",               signal="airspeed / wind"),
    "wing_cs":   dict(subclass=["wing", "wing bristle", "campaniform sensilla"],
                      signal="wing strain and load"),
    "hairplate": dict(subclass=["hair plate", "neck"], signal="joint angle / head-body angle"),
    "chordotonal": dict(subclass="chordotonal organ", signal="limb proprioception"),
}


@dataclass
class FlightCircuit:
    """A connectome subgraph partitioned into afferent / intrinsic / efferent."""

    cx: Connectome
    afferent: dict[str, np.ndarray]          # port name -> row indices (local to cx)
    efferent: dict[str, np.ndarray]          # motor group -> row indices
    intrinsic: np.ndarray                    # everything else
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
        """+1 right, -1 left, 0 unknown/midline — needed for asymmetric steering."""
        s = self.cx.meta.somaSide.to_numpy()[rows]
        return np.where(s == "R", 1, np.where(s == "L", -1, 0))

    def summary(self) -> str:
        lines = [self.cx.summary(), ""]
        lines.append("  afferent ports:")
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


def _khop(W: sp.csr_matrix, seeds: np.ndarray, hops: int, reverse: bool) -> np.ndarray:
    """Reachable set within `hops` synapses. W[post, pre], so forward = W @ x."""
    n = W.shape[0]
    A = (W.T if reverse else W)
    A = sp.csr_matrix((np.ones_like(A.data), A.indices, A.indptr), shape=A.shape)
    x = np.zeros(n, dtype=bool)
    x[seeds] = True
    reached = x.copy()
    for _ in range(hops):
        x = (A @ x.astype(np.float32)) > 0
        new = x & ~reached
        if not new.any():
            break
        reached |= new
    return np.flatnonzero(reached)


def _prune_to_budget(cx, rows, afferent, eff_all, budget, verbose):
    """Shrink the circuit to `budget` neurons without destroying its structure.

    Naively keeping every seed does not work: the retinotopic input alone is
    14,201 neurons, so a small budget would be entirely eye and no brain. The
    order of protection is therefore:

      1. every motor neuron — there are only ~430 and each one is a distinct
         muscle, so losing any loses a degree of freedom outright;
      2. the non-visual sensors, which are small and irreplaceable;
      3. the retina, subsampled *by ommatidial column* rather than by neuron, so
         the eye keeps a coherent (if coarser) map of visual space instead of a
         scatter of blind spots;
      4. whatever budget is left goes to the interneurons with the highest
         synaptic throughput inside the subgraph.
    """
    rng = np.random.default_rng(0)
    keep = set(int(i) for i in eff_all)

    small = np.concatenate([v for k, v in afferent.items() if k != "retina" and len(v)]) \
        if any(k != "retina" and len(v) for k, v in afferent.items()) else np.array([], np.int64)
    keep |= set(int(i) for i in small)

    retina = afferent.get("retina", np.array([], np.int64))
    if len(retina):
        share = int(0.30 * budget)          # the eye gets 30% of the budget
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
    hops: int = 2,
    max_neurons: int | None = None,
    verbose: bool = True,
) -> FlightCircuit:
    """Extract the sensorimotor circuit for flight.

    scale:
      ``"full"``    every traced neuron in the CNS (~165 k) — the honest maximum.
      ``"flight"``  neurons on a <=`hops`-synapse path from a flight sensor to a
                    flight motor neuron.  This is the default: it is the tissue
                    that can actually influence a wing within the latency budget
                    of a wingbeat.
      ``"core"``    the flight subgraph pruned to its highest-throughput core,
                    for fast iteration on a laptop.
    """
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
        fwd = _khop(cx.W, aff_all, hops, reverse=False)     # downstream of sensors
        bwd = _khop(cx.W, eff_all, hops, reverse=True)      # upstream of muscles
        between = np.intersect1d(fwd, bwd)
        rows = np.union1d(np.union1d(aff_all, eff_all), between)
        if verbose:
            print(f"[circuit] {len(fwd):,} within {hops} hops of a sensor, "
                  f"{len(bwd):,} within {hops} hops of a muscle, "
                  f"{len(between):,} on a path between")

    if scale == "core" or (max_neurons and len(rows) > max_neurons):
        budget = max_neurons or 6000
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

    # retinotopic address for the visual port, so the renderer knows where each
    # ommatidium's output belongs
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
