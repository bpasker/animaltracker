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

    @property
    def frame_count(self) -> int:
        """Current number of frames in the buffer."""
        return len(self._buffer)

    @property
    def max_frames(self) -> int:
        """Maximum buffer capacity in frames."""
        return int(self.max_seconds * self.fps)

    @property
    def duration(self) -> float:
        """Current buffer duration in seconds based on frame count."""
        return len(self._buffer) / self.fps if self.fps > 0 else 0.0

    def push(self, timestamp: float, frame: np.ndarray) -> None:
        with self._lock:
            self._buffer.append((timestamp, frame.copy()))

    def dump(self) -> List[FramePayload]:
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
