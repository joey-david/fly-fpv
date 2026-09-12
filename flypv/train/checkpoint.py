"""Atomic, resumable training checkpoints with retention and best/latest pointers."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAT_VERSION = 2


class CheckpointManager:
    def __init__(self, run_dir: str | Path, keep: int = 5, best_metric: str = "ep_gates"):
        self.run_dir = Path(run_dir)
        self.directory = self.run_dir / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep = max(1, int(keep))
        self.best_metric = best_metric

    @staticmethod
    def resolve(run_dir: str | Path, spec: str | Path) -> Path:
        p = Path(spec)
        if p.exists():
            return p.resolve()
        run_dir = Path(run_dir)
        if str(spec) in {"latest", "best"}:
            manifest = run_dir / "checkpoints" / f"{spec}.json"
            if not manifest.exists():
                raise FileNotFoundError(f"no {spec} checkpoint for run {run_dir.name!r}")
            meta = json.loads(manifest.read_text())
            target = run_dir / "checkpoints" / meta["file"]
            if not target.exists():
                raise FileNotFoundError(f"checkpoint pointer is stale: {target}")
            return target.resolve()
        candidate = run_dir / "checkpoints" / str(spec)
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"checkpoint not found: {spec}")

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if payload.get("format_version", 1) > FORMAT_VERSION:
            raise RuntimeError(
                f"checkpoint format {payload['format_version']} is newer than supported {FORMAT_VERSION}"
            )
        return payload

    def save(
        self,
        *,
        policy,
        optimizer,
        cfg: dict[str, Any],
        global_step: int,
        updates: int,
        difficulty: float,
        ep_stats: dict[str, list[float]],
        metrics: dict[str, float],
        elapsed: float,
        reason: str = "periodic",
    ) -> Path:
        path = self.directory / f"step_{global_step:09d}.pt"
        payload = {
            "format_version": FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(global_step),
            "updates": int(updates),
            "difficulty": float(difficulty),
            "elapsed": float(elapsed),
            "ep_stats": ep_stats,
            "metrics": metrics,
            "cfg": cfg,
            "rng": self._rng_state(),
        }
        self._atomic_torch_save(payload, path)
        self._write_pointer("latest", path, metrics)

        current = metrics.get(self.best_metric)
        best = self._read_pointer("best")
        best_value = None if best is None else best.get("metric_value")
        if current is not None and (best_value is None or float(current) > float(best_value)):
            self._write_pointer("best", path, metrics, metric_value=float(current))

        self._prune()
        return path

    def _write_pointer(
        self,
        name: str,
        path: Path,
        metrics: dict[str, float],
        metric_value: float | None = None,
    ) -> None:
        data = {
            "file": path.name,
            "step": int(path.stem.split("_")[-1]),
            "metric": self.best_metric if name == "best" else None,
            "metric_value": metric_value,
            "metrics": metrics,
        }
        self._atomic_text(json.dumps(data, indent=2) + "\n", self.directory / f"{name}.json")

    def _read_pointer(self, name: str) -> dict[str, Any] | None:
        p = self.directory / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _prune(self) -> None:
        protected = set()
        for name in ("latest", "best"):
            p = self._read_pointer(name)
            if p:
                protected.add(p["file"])
        checkpoints = sorted(self.directory.glob("step_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        keep_names = {p.name for p in checkpoints[: self.keep]} | protected
        for p in checkpoints:
            if p.name not in keep_names:
                p.unlink(missing_ok=True)

    @staticmethod
    def restore_rng(payload: dict[str, Any]) -> None:
        rng = payload.get("rng") or {}
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        return {
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @staticmethod
    def _atomic_text(text: str, path: Path) -> None:
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
