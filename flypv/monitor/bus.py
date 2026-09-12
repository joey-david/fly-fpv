"""A single-writer telemetry bus between the trainer and the dashboard.

The trainer must never block on, or be slowed by, the monitor. So every channel
here is latest-value-wins: the trainer overwrites, the websocket reads whatever
is current at its own pace, and a slow browser simply drops frames.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


class Bus:
    def __init__(self, history: int = 3000):
        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {}
        self._static: dict[str, Any] = {}
        self._series: dict[str, deque] = {}
        self._history = history
        self.t0 = time.time()
        self.seq = 0

    # --- static payloads sent once when a client connects ------------------
    def set_static(self, key: str, value: Any):
        with self._lock:
            self._static[key] = value

    def static(self) -> dict:
        with self._lock:
            return dict(self._static)

    # --- live frames -------------------------------------------------------
    def publish(self, channel: str, value: Any):
        with self._lock:
            self._latest[channel] = value
            self.seq += 1

    def record(self, **scalars):
        """Append scalars to their time series and to the latest frame."""
        with self._lock:
            for k, v in scalars.items():
                if k not in self._series:
                    self._series[k] = deque(maxlen=self._history)
                self._series[k].append(float(v))
            self._latest.setdefault("metrics", {}).update(
                {k: float(v) for k, v in scalars.items()})
            self.seq += 1

    def snapshot(self) -> dict:
        with self._lock:
            out = {k: v for k, v in self._latest.items()}
            out["t"] = time.time() - self.t0
            out["seq"] = self.seq
            return out

    def series(self, keys: list[str] | None = None, stride: int = 1) -> dict:
        with self._lock:
            keys = keys or list(self._series)
            return {k: list(self._series[k])[::stride] for k in keys if k in self._series}


BUS = Bus()
