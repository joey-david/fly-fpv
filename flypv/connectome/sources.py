"""Download endpoints for the Drosophila connectome releases we build on.

Primary dataset: **MaleCNS v1.0** (Janelia FlyEM + Cambridge Drosophila
Connectomics + Google Research, CC-BY).  It is the first complete brain *and*
ventral nerve cord of an adult fly — 166k+ neurons — which is what makes it the
right substrate here: the brain alone cannot flap a wing.  The wing motor
neurons and the flight power/steering circuits live in the VNC half.

Everything is served as Apache Arrow Feather from a public GCS bucket, no
credentials required.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = Path(os.environ.get("FLYPV_DATA", REPO_ROOT / "data" / "raw"))
DATA_CACHE = Path(os.environ.get("FLYPV_CACHE", REPO_ROOT / "data" / "cache"))

_MCNS = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"


@dataclass(frozen=True)
class Source:
    key: str
    filename: str
    url: str
    size_mb: int
    tier: str  # "core" = needed to train; "extra" = optional analysis/visuals
    what: str


SOURCES: dict[str, Source] = {
    s.key: s
    for s in [
        Source(
            "annotations",
            "body-annotations-male-cns-v1.0-minconf-0.5.feather",
            f"{_MCNS}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
            13,
            "core",
            "Per-neuron curated labels: cell type, class, superclass, side, neuromere. "
            "This is how we know which neurons are photoreceptors and which drive wings.",
        ),
        Source(
            "neurotransmitters",
            "body-neurotransmitters-male-cns-v1.0.feather",
            f"{_MCNS}/body-neurotransmitters-male-cns-v1.0.feather",
            42,
            "core",
            "Predicted neurotransmitter per neuron. Gives every synapse a SIGN "
            "(ACh/Glu/His excitatory, GABA/Gly inhibitory) instead of a free parameter.",
        ),
        Source(
            "weights",
            "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
            f"{_MCNS}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
            1100,
            "core",
            "The wiring diagram itself: bodyId_pre -> bodyId_post with synapse counts.",
        ),
        Source(
            "body_stats",
            "body-stats-male-cns-v1.0-minconf-0.5.feather",
            f"{_MCNS}/body-stats-male-cns-v1.0-minconf-0.5.feather",
            780,
            "extra",
            "Per-segment synapse counts and ROI breakdown. Used for neuropil-level layout.",
        ),
        Source(
            "syn_partners",
            "syn-partners-male-cns-v1.0-minconf-0.5.feather",
            f"{_MCNS}/syn-partners-male-cns-v1.0-minconf-0.5.feather",
            6800,
            "extra",
            "Every synaptic pair with its neuropil. Only needed for ROI-resolved analysis.",
        ),
    ]
}

LICENSE_NOTE = (
    "MaleCNS v1.0 is CC-BY. Acquired, reconstructed and annotated by the Janelia FlyEM "
    "Project Team and the Cambridge Drosophila Connectomics Group, in collaboration with "
    "the Connectomics group at Google Research. Cite them if you publish anything from this."
)


def curl_commands(tier: str = "core") -> list[str]:
    """The plain download lines, if you would rather run them yourself."""
    out = []
    for s in SOURCES.values():
        if tier != "all" and s.tier != tier:
            continue
        out.append(f'curl -L -C - -o "{DATA_RAW / s.filename}" "{s.url}"')
    return out


def path_of(key: str) -> Path:
    return DATA_RAW / SOURCES[key].filename


def have(key: str) -> bool:
    p = path_of(key)
    # a truncated/resumed download is smaller than ~80% of the advertised size
    return p.exists() and p.stat().st_size > 0.8 * SOURCES[key].size_mb * 1e6


def fetch(keys: list[str] | None = None, tier: str = "core", force: bool = False) -> list[Path]:
    """Download the requested sources with resume support. Returns local paths."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if keys is None:
        keys = [k for k, s in SOURCES.items() if tier == "all" or s.tier == tier]

    paths = []
    for k in keys:
        s = SOURCES[k]
        dest = DATA_RAW / s.filename
        if have(k) and not force:
            print(f"  [have] {s.filename}")
            paths.append(dest)
            continue
        print(f"  [get ] {s.filename}  (~{s.size_mb} MB)  {s.what.splitlines()[0]}")
        cmd = ["curl", "-L", "-C", "-", "--retry", "3", "-#", "-o", str(dest), s.url]
        subprocess.run(cmd, check=True)
        paths.append(dest)
    return paths
