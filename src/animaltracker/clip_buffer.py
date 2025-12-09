"""Circular clip buffer utilities."""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np

FramePayload = Tuple[float, np.ndarray]


@dataclass
class ClipBuffer:
    max_seconds: float
    fps: float

    def __post_init__(self) -> None:
        self._buffer: Deque[FramePayload] = deque(maxlen=int(self.max_seconds * self.fps))
        self._lock = threading.Lock()

    def push(self, timestamp: float, frame: np.ndarray) -> None:
        with self._lock:
            self._buffer.append((timestamp, frame.copy()))

    def dump(self) -> List[FramePayload]:
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
