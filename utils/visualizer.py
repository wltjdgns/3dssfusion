from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from models.base import DetectionResult

_PALETTE = [
    (0, 255, 0),   (255, 80, 0),  (0, 80, 255),  (255, 255, 0),
    (0, 255, 255), (255, 0, 200), (128, 255, 0),  (255, 160, 0),
    (0, 160, 255), (200, 0, 255), (0, 255, 160),  (255, 0, 80),
]


def _color(class_id: int) -> tuple:
    return _PALETTE[class_id % len(_PALETTE)]


import os as _os
_PROC = None

def _get_system_stats() -> dict:
    """이 프로세스의 CPU%, GPU%, VRAM 반환. GPU%는 시스템 전체 (per-process API 불안정)."""
    global _PROC
    stats: dict = {}
    try:
        import psutil
        if _PROC is None:
            _PROC = psutil.Process(_os.getpid())
        # cpu_percent(interval=None): 마지막 호출 이후 이 프로세스의 CPU 사용률
        # num_cpus=True → 전체 코어 대비 % (예: 4코어 100% 사용 시 400%)
        # / cpu_count() 하면 전체 대비 % 로 환산
        cpu_raw = _PROC.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count(logical=True) or 1
        stats["cpu"] = cpu_raw / cpu_count   # 전체 코어 대비 %
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            # memory_allocated: 이 프로세스 PyTorch 텐서 사용량 (per-process ✓)
            stats["vram_used"] = torch.cuda.memory_allocated(dev) / 1024 ** 2
            stats["vram_total"] = torch.cuda.get_device_properties(dev).total_memory / 1024 ** 2
            # utilization: 시스템 전체 GPU % (per-process API 드라이버 의존적이라 생략)
            stats["gpu"] = torch.cuda.utilization(dev)
    except Exception:
        pass
    return stats


def draw_detections(
    frame: np.ndarray,
    result: DetectionResult,
    font_scale: float = 0.55,
    thickness: int = 2,
) -> np.ndarray:
    out = frame.copy()
    for det in result.detections:
        x1, y1, x2, y2 = det.bbox_2d.astype(int)
        color = _color(det.class_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.class_name} {det.score:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - bl - 2), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - bl - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _depth_colormap(depth: np.ndarray, hw: tuple) -> np.ndarray:
    d8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    h, w = hw
    return cv2.resize(colored, (w, h))


def _draw_overlay(vis: np.ndarray, fps: float, show_fps: bool, show_gpu: bool) -> np.ndarray:
    """FPS / GPU / CPU 오버레이를 왼쪽 상단에 그립니다."""
    lines = []
    if show_fps:
        lines.append(f"FPS: {fps:.1f}")
    if show_gpu:
        stats = _get_system_stats()
        if "gpu" in stats:
            lines.append(f"GPU(sys): {stats['gpu']}%  VRAM(proc): {stats['vram_used']:.0f}/{stats['vram_total']:.0f}MB")
        if "cpu" in stats:
            lines.append(f"CPU(proc): {stats['cpu']:.1f}%")

    fs = 0.65
    th = 1
    for i, line in enumerate(lines):
        y = 28 + i * 26
        (tw, lh), bl = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.rectangle(vis, (8, y - lh - 2), (8 + tw + 4, y + bl), (0, 0, 0), -1)
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), th, cv2.LINE_AA)
    return vis


class Visualizer:
    """OpenCV 기반 실시간 탐지 결과 시각화."""

    def __init__(self, config: dict) -> None:
        self._enabled: bool = config.get("show_window", True)
        self._window: str = config.get("window_name", "Kinect Detection")
        self._show_depth: bool = config.get("show_depth_colormap", True)
        self._show_fps: bool = config.get("show_fps", True)
        self._show_gpu: bool = config.get("show_gpu_usage", True)
        self._font_scale: float = config.get("font_scale", 0.6)
        self._thickness: int = config.get("line_thickness", 2)

    def render(
        self,
        color: np.ndarray,
        result: DetectionResult,
        depth: Optional[np.ndarray] = None,
        fps: float = 0.0,
    ) -> bool:
        """프레임을 렌더링합니다. q 누르면 False 반환 (종료 신호)."""
        if not self._enabled or color is None:
            return True

        vis = draw_detections(color, result, self._font_scale, self._thickness)

        if self._show_depth and depth is not None:
            h, w = vis.shape[:2]
            half_w = w // 2
            half_h = int(h * half_w / w)
            color_half = cv2.resize(vis, (half_w, half_h))
            depth_half = _depth_colormap(depth, (half_h, half_w))
            vis = np.hstack([color_half, depth_half])

        _draw_overlay(vis, fps, self._show_fps, self._show_gpu)

        cv2.imshow(self._window, vis)
        return (cv2.waitKey(1) & 0xFF) != ord('q')

    def close(self) -> None:
        if self._enabled:
            cv2.destroyAllWindows()
