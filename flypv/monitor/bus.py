"""A single-writer telemetry bus between the trainer and the dashboard.

Channels are latest-value-wins. Each channel also tracks the sequence at which it
last changed so websocket clients can receive deltas instead of the entire live
state every frame. That matters because neural activity is much larger than the
flight pose and should not be re-encoded/reparsed at video rate when it has not
changed.
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
        self._channel_seq: dict[str, int] = {}
        self._static: dict[str, Any] = {}
        self._series: dict[str, deque] = {}
        self._history = history
        self.t0 = time.time()
        self.seq = 0

    def set_static(self, key: str, value: Any):
        with self._lock:
            self._static[key] = value

    def static(self) -> dict:
        with self._lock:
            return dict(self._static)

    def publish(self, channel: str, value: Any):
        with self._lock:
            self.seq += 1
            self._latest[channel] = value
            self._channel_seq[channel] = self.seq

    def publish_many(self, **channels: Any):
        """Atomically replace several live channels with one sequence number."""
        if not channels:
            return
        with self._lock:
            self.seq += 1
            self._latest.update(channels)
            for key in channels:
                self._channel_seq[key] = self.seq

    def record(self, **scalars):
        """Append scalar histories and publish the newest metric values."""
        with self._lock:
            for key, value in scalars.items():
                if key not in self._series:
                    self._series[key] = deque(maxlen=self._history)
                self._series[key].append(float(value))
            metrics = self._latest.setdefault("metrics", {})
            metrics.update({key: float(value) for key, value in scalars.items()})
            self.seq += 1
            self._channel_seq["metrics"] = self.seq

    def snapshot(self, since: int | None = None) -> dict:
        """Return the latest values, optionally only channels changed after ``since``."""
        with self._lock:
            if since is None:
                out = dict(self._latest)
            else:
                out = {
                    key: value
                    for key, value in self._latest.items()
                    if self._channel_seq.get(key, 0) > since
                }
            out["t"] = time.time() - self.t0
            out["seq"] = self.seq
            return out

    def series(self, keys: list[str] | None = None, stride: int = 1) -> dict:
        with self._lock:
            keys = keys or list(self._series)
            return {key: list(self._series[key])[::stride] for key in keys if key in self._series}


BUS = Bus()
