"""Turn the MaleCNS feather dumps into a signed sparse synaptic weight matrix.

The headline number for MaleCNS v1.0 is "166,000 neurons", but the raw
`connectome-weights` table has 151.8 M rows because it is segment-to-segment,
including every unproofread fragment. Training on that unfiltered is training
on reconstruction noise. We filter twice:

  1. **Bodies** — keep proofread neurons (`status == "Traced"`), drop glia and
     orphan fragments.
  2. **Edges** — keep connections at or above `min_synapses` (default 5). This
     is the conventional connectome-analysis threshold: below it the false
     discovery rate from automatic synapse prediction dominates.

Every surviving edge then gets a *sign* from the presynaptic neuron's predicted
neurotransmitter, so the network's excitation/inhibition structure is measured
biology rather than something gradient descent has to invent.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scipy.sparse as sp

from .sources import DATA_CACHE, SOURCES, have, path_of

# Sign of each neurotransmitter's fast ionotropic action in Drosophila.
# Histamine is the photoreceptor transmitter and is inhibitory onto LMCs —
# that sign inversion is why L1/L2 respond to contrast decrements.
NT_SIGN: dict[str, float] = {
    "acetylcholine": +1.0,
    "glutamate": -1.0,   # Drosophila GluCl-alpha is the common postsynaptic receptor
    "gaba": -1.0,
    "glycine": -1.0,
    "histamine": -1.0,
    "dopamine": 0.0,     # modulatory: no fast sign, treated as gain-only
    "octopamine": 0.0,
    "serotonin": 0.0,
    "unclear": 0.0,      # resolved to a learned sign at model build time
}

ANNOT_COLS = [
    "bodyId", "type", "class", "subclass", "superclass", "somaNeuromere",
    "somaSide", "status", "entryNerve", "exitNerve",
    "assignedOlHex1", "assignedOlHex2", "instance", "group", "somaLocation",
]


@dataclass
class Connectome:
    """A filtered, signed, index-compacted connectome."""

    body_ids: np.ndarray            # (N,) int64 — MaleCNS bodyIds, our canonical order
    W: sp.csr_matrix                # (N, N) float32 — W[post, pre] = signed synapse count
    meta: pd.DataFrame              # (N, ...) annotations, row-aligned to body_ids
    nt_sign: np.ndarray             # (N,) float32 — presynaptic sign per neuron
    source: str = "male-cns-v1.0"
    min_synapses: int = 5
    _index: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self._index:
            self._index = {int(b): i for i, b in enumerate(self.body_ids)}

    @property
    def n(self) -> int:
        return len(self.body_ids)

    @property
    def n_synapses(self) -> int:
        return int(np.abs(self.W.data).sum())

    def idx(self, body_ids) -> np.ndarray:
        """Map bodyIds -> row indices, silently dropping ones we filtered out."""
        return np.array([self._index[int(b)] for b in body_ids if int(b) in self._index], dtype=np.int64)

    def select(self, **kw) -> np.ndarray:
        """Row indices whose metadata matches. Values may be scalars, lists, or regex via `type_re`."""
        m = pd.Series(True, index=self.meta.index)
        for k, v in kw.items():
            if k.endswith("_re"):
                col = k[:-3]
                m &= self.meta[col].fillna("").str.contains(v, regex=True)
            elif isinstance(v, (list, tuple, set)):
                m &= self.meta[k].isin(list(v))
            else:
                m &= self.meta[k] == v
        return np.flatnonzero(m.to_numpy())

    def subgraph(self, rows: np.ndarray) -> "Connectome":
        rows = np.unique(np.asarray(rows, dtype=np.int64))
        W = self.W[rows][:, rows].tocsr()
        meta = self.meta.iloc[rows].reset_index(drop=True)
        return Connectome(
            body_ids=self.body_ids[rows], W=W, meta=meta, nt_sign=self.nt_sign[rows],
            source=self.source, min_synapses=self.min_synapses,
        )

    def summary(self) -> str:
        sc = self.meta.superclass.fillna("?").value_counts()
        exc = float((self.W.data > 0).mean()) if self.W.nnz else 0.0
        return (
            f"{self.source}: {self.n:,} neurons, {self.W.nnz:,} connections, "
            f"{self.n_synapses:,} synapses, {exc:.0%} excitatory\n"
            + "\n".join(f"    {k:<22} {v:>7,}" for k, v in sc.head(12).items())
        )

    # ---- persistence -------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            body_ids=self.body_ids, nt_sign=self.nt_sign,
            data=self.W.data, indices=self.W.indices, indptr=self.W.indptr, shape=self.W.shape,
            info=json.dumps({"source": self.source, "min_synapses": self.min_synapses}),
        )
        self.meta.to_parquet(path.with_suffix(".meta.parquet"))
        return path

    @classmethod
    def load(cls, path: Path) -> "Connectome":
        path = Path(path)
        z = np.load(path, allow_pickle=False)
        info = json.loads(str(z["info"]))
        W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        meta = pd.read_parquet(path.with_suffix(".meta.parquet"))
        return cls(body_ids=z["body_ids"], W=W, meta=meta, nt_sign=z["nt_sign"], **info)


def _load_annotations() -> pd.DataFrame:
    t = feather.read_table(path_of("annotations"), columns=ANNOT_COLS, memory_map=True)
    d = t.to_pandas()
    for c in ("type", "class", "subclass", "superclass", "somaNeuromere", "somaSide",
              "entryNerve", "exitNerve", "instance"):
        d[c] = d[c].fillna("").astype(str)

    # Soma coordinates come as a length-3 list of 8 nm voxel indices. Unpack them
    # into plain columns (in micrometres) so the visualiser can lay neurons out
    # where they actually sit in the animal rather than by force-directed guess.
    loc = d.pop("somaLocation")
    xyz = np.full((len(d), 3), np.nan, dtype=np.float32)
    ok = loc.notna().to_numpy()
    if ok.any():
        arr = np.array([v if isinstance(v, (list, np.ndarray)) and len(v) == 3 else (np.nan,) * 3
                        for v in loc[ok]], dtype=np.float64)
        xyz[ok] = (arr * 8.0 / 1000.0).astype(np.float32)
    d["soma_x"], d["soma_y"], d["soma_z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return d


def _load_nt(body_ids: np.ndarray) -> np.ndarray:
    """Per-neuron presynaptic sign, preferring the cell-type consensus call.

    `consensus_nt` is "unclear" for the vast majority of the 1.8 M *segments*,
    but resolves for most genuinely traced neurons; where it doesn't we fall
    back to the cell-type-level prediction, which pools evidence across all
    cells of that type and is far better powered than a single fragment.
    """
    t = feather.read_table(
        path_of("neurotransmitters"),
        columns=["body", "consensus_nt", "celltype_predicted_nt", "predicted_nt"],
        memory_map=True,
    ).to_pandas()
    t = t.set_index("body")
    t = t.reindex(body_ids)

    nt = t["consensus_nt"].fillna("unclear").replace("unclear", np.nan)
    nt = nt.fillna(t["celltype_predicted_nt"]).fillna(t["predicted_nt"]).fillna("unclear")
    return nt.map(lambda s: NT_SIGN.get(str(s).lower(), 0.0)).to_numpy(dtype=np.float32)


def _repair_metadata_sidecar(cache_path: Path, verbose: bool = True) -> bool:
    """Recreate a missing metadata parquet from the raw annotation table.

    The sparse cache stores the exact body-id ordering, so rebuilding this sidecar
    does not require rescanning the 151.8 M-edge weights table.
    """
    meta_path = cache_path.with_suffix(".meta.parquet")
    if not cache_path.exists() or meta_path.exists() or not have("annotations"):
        return False

    if verbose:
        print(f"[connectome] cache metadata missing; repairing {meta_path.name} ...")
    with np.load(cache_path, allow_pickle=False) as z:
        body_ids = z["body_ids"].astype(np.int64, copy=False)

    ann = _load_annotations().set_index("bodyId", drop=False)
    missing = np.setdiff1d(body_ids, ann.index.to_numpy(dtype=np.int64), assume_unique=False)
    if len(missing):
        if verbose:
            print(f"[connectome] cannot repair metadata: {len(missing):,} cached bodies are absent from annotations")
        return False

    meta = ann.loc[body_ids].reset_index(drop=True)
    meta.to_parquet(meta_path)
    if verbose:
        print(f"[connectome] repaired -> {meta_path.name}")
    return True


def load_connectome(
    min_synapses: int = 5,
    statuses: tuple[str, ...] = ("Traced",),
    require_type: bool = False,
    cache: bool = True,
    verbose: bool = True,
    force_rebuild: bool = False,
) -> Connectome:
    """Build (or load from cache) the filtered whole-CNS connectome."""
    tag = f"malecns-v1.0-syn{min_synapses}-{'typed' if require_type else 'all'}"
    cache_path = DATA_CACHE / f"{tag}.npz"
    meta_path = cache_path.with_suffix(".meta.parquet")

    if cache and not force_rebuild and cache_path.exists():
        if not meta_path.exists():
            _repair_metadata_sidecar(cache_path, verbose=verbose)
        if meta_path.exists():
            if verbose:
                print(f"[connectome] cache hit -> {cache_path.name}")
            try:
                return Connectome.load(cache_path)
            except (OSError, ValueError, EOFError, KeyError) as exc:
                if verbose:
                    print(f"[connectome] cache is invalid ({exc}); rebuilding")
        elif verbose:
            print("[connectome] incomplete cache; rebuilding from raw data")
    elif cache and force_rebuild and verbose:
        print(f"[connectome] forcing rebuild of {cache_path.name}")

    for k in ("annotations", "neurotransmitters", "weights"):
        if not have(k):
            raise FileNotFoundError(
                f"missing {SOURCES[k].filename}. Run:  flypv fetch   "
                f"(or see flypv.connectome.sources.curl_commands())"
            )

    t0 = time.time()
    ann = _load_annotations()
    keep = ann.status.isin(statuses)
    if require_type:
        keep &= ann["type"].str.len() > 0
    ann = ann[keep].reset_index(drop=True)
    body_ids = ann.bodyId.to_numpy(dtype=np.int64)
    if verbose:
        print(f"[connectome] {len(body_ids):,} bodies after status filter {statuses}")

    nt_sign = _load_nt(body_ids)
    if verbose:
        known = float((nt_sign != 0).mean())
        print(f"[connectome] neurotransmitter sign resolved for {known:.1%} of neurons")

    # The weights table is 151.8 M rows / ~3.6 GB in memory as int64 triples.
    # Read it as Arrow and filter in chunks so peak RSS stays modest.
    keep_set = pd.Index(body_ids)
    tbl = feather.read_table(path_of("weights"), memory_map=True)
    if verbose:
        print(f"[connectome] filtering {tbl.num_rows:,} segment-level edges ...")

    pre_parts, post_parts, w_parts = [], [], []
    n_rows = tbl.num_rows
    chunk = 8_000_000
    for start in range(0, n_rows, chunk):
        sl = tbl.slice(start, chunk)
        w = sl.column("weight").to_numpy()
        m = w >= min_synapses
        if not m.any():
            continue
        pre = sl.column("body_pre").to_numpy()[m]
        post = sl.column("body_post").to_numpy()[m]
        w = w[m]
        ip = keep_set.get_indexer(pre)
        iq = keep_set.get_indexer(post)
        ok = (ip >= 0) & (iq >= 0)
        if ok.any():
            pre_parts.append(ip[ok]); post_parts.append(iq[ok]); w_parts.append(w[ok])
    del tbl

    ip = np.concatenate(pre_parts); iq = np.concatenate(post_parts)
    w = np.concatenate(w_parts).astype(np.float32)
    del pre_parts, post_parts, w_parts

    # Signed by the PRESYNAPTIC neuron's transmitter. Unresolved neurons keep a
    # +1 magnitude here; the policy gives them a learnable sign at build time.
    signs = np.where(nt_sign[ip] == 0, 1.0, nt_sign[ip]).astype(np.float32)
    W = sp.csr_matrix((w * signs, (iq, ip)), shape=(len(body_ids), len(body_ids)), dtype=np.float32)
    W.sum_duplicates()

    c = Connectome(body_ids=body_ids, W=W, meta=ann, nt_sign=nt_sign, min_synapses=min_synapses)
    if verbose:
        print(f"[connectome] built in {time.time()-t0:.1f}s\n{c.summary()}")
    if cache:
        c.save(cache_path)
        if verbose:
            print(f"[connectome] cached -> {cache_path}")
    return c
