from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from loguru import logger

from models.base import DetectionResult
from kinect.capture import CaptureFrame


_SOURCE_COLORS: dict[str, tuple[int, int, int]] = {
    "rgb":    (0, 255, 0),    # 초록
    "depth":  (255, 0, 0),    # 파랑
    "fusion": (0, 0, 255),    # 빨강
}
_DEFAULT_COLOR = (200, 200, 200)


class ResultVisualizer:
    """OpenCV 기반 실시간 탐지 결과 시각화."""

    def __init__(self, config: dict) -> None:
        self._show_window: bool = config.get("show_window", True)
        self._window_name: str = config.get("window_name", "Kinect Detection")
        self._show_depth_colormap: bool = config.get("show_depth_colormap", True)
        self._show_fps: bool = config.get("show_fps", True)
        self._show_gpu_usage: bool = config.get("show_gpu_usage", False)
        self._font_scale: float = float(config.get("font_scale", 0.6))
        self._thickness: int = int(config.get("line_thickness", 2))

        self._writer: Optional[cv2.VideoWriter] = None
        self._pending_output_path: Optional[str] = None
        save_video: bool = config.get("save_video", False)
        output_path: str = config.get("output_path", "data/results/output.avi")
        if save_video:
            self._init_writer(output_path)

    def draw(
        self,
        frame: CaptureFrame,
        result: DetectionResult,
        fps: float,
        gpu_usage: Optional[dict] = None,
    ) -> np.ndarray:
        """탐지 결과를 프레임에 그려 BGR 이미지를 반환합니다."""
        canvas = frame.color.copy()

        for det in result.detections:
            x1, y1, x2, y2 = det.bbox_2d.astype(int)
            color = _SOURCE_COLORS.get(det.source, _DEFAULT_COLOR)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self._thickness)

            label = self._build_label(det)
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, self._thickness
            )
            ly = max(y1 - 4, th + baseline)
            cv2.rectangle(canvas, (x1, ly - th - baseline), (x1 + tw, ly + baseline), color, cv2.FILLED)
            cv2.putText(
                canvas, label, (x1, ly),
                cv2.FONT_HERSHEY_SIMPLEX, self._font_scale,
                (0, 0, 0), self._thickness, cv2.LINE_AA,
            )

        if self._show_depth_colormap:
            canvas = self._overlay_depth_colormap(canvas, frame.depth)

        if self._show_fps:
            cv2.putText(
                canvas, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, self._font_scale + 0.2,
                (0, 255, 255), self._thickness, cv2.LINE_AA,
            )

        if self._show_gpu_usage and gpu_usage:
            used = gpu_usage.get("vram_used_mb", 0)
            total = gpu_usage.get("vram_total_mb", 0)
            util = gpu_usage.get("utilization_pct", 0)
            text = f"VRAM: {used}/{total}MB  GPU: {util}%"
            cv2.putText(
                canvas, text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, self._font_scale,
                (0, 200, 255), self._thickness, cv2.LINE_AA,
            )

        if self._pending_output_path is not None:
            self._ensure_writer(canvas.shape)
        if self._writer is not None:
            self._writer.write(canvas)

        return canvas

    def show(self, image: np.ndarray) -> bool:
        """이미지를 윈도우에 표시합니다. ESC 입력 시 False를 반환합니다."""
        if not self._show_window:
            return True
        cv2.imshow(self._window_name, image)
        key = cv2.waitKey(1) & 0xFF
        return key != 27  # ESC

    def release(self) -> None:
        """VideoWriter 및 윈도우 리소스를 해제합니다."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("VideoWriter 해제 완료.")
        if self._show_window:
            cv2.destroyAllWindows()

    def _build_label(self, det) -> str:
        parts: list[str] = []
        if det.track_id is not None:
            parts.append(f"[{det.track_id}]")
        parts.append(det.class_name or str(det.class_id))
        parts.append(f"{det.score:.2f}")
        if det.depth_m is not None:
            parts.append(f"{det.depth_m:.2f}m")
        return " ".join(parts)

    def _overlay_depth_colormap(
        self,
        canvas: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        """depth 이미지를 colormap으로 변환하여 우상단에 오버레이합니다."""
        h, w = canvas.shape[:2]
        overlay_h = h // 4
        overlay_w = w // 4

        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        depth_resized = cv2.resize(depth_color, (overlay_w, overlay_h))

        canvas[0:overlay_h, w - overlay_w:w] = depth_resized
        return canvas

    def _init_writer(self, output_path: str) -> None:
        """VideoWriter 경로를 등록합니다. 실제 초기화는 첫 프레임 수신 시 수행합니다."""
        import os
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        self._pending_output_path = output_path

    def _ensure_writer(self, frame_shape: tuple) -> None:
        """첫 프레임 도달 시 실제 해상도로 VideoWriter를 초기화합니다."""
        if self._writer is not None:
            return
        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        path = getattr(self, "_pending_output_path", "data/results/output.avi")
        self._writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
        if not self._writer.isOpened():
            logger.warning(f"VideoWriter 초기화 실패: {path}")
            self._writer = None
        else:
            logger.info(f"VideoWriter 초기화: {path} ({w}x{h})")
