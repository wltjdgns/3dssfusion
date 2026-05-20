from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Rolling average 방식 FPS 측정기."""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        """현재 타임스탬프를 기록하고 현재 FPS를 반환합니다."""
        self._timestamps.append(time.perf_counter())
        return self.fps

    @property
    def fps(self) -> float:
        """최근 window_size 프레임 기반 rolling average FPS."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
