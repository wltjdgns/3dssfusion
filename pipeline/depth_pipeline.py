"""
DepthPipeline: Depth(PointCloud) 단독 탐지 모드
- KinectCapture → PointCloudConverter → PointPillarsDetector
- 3D bbox를 project_3d_to_2d로 2D 투영 후 DetectionResult에 포함
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Optional

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from models import DetectionResult
    from kinect import KinectCapture, KinectCalibration, PointCloudConverter


class DepthPipeline:
    """Depth 단독 탐지 파이프라인."""

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._depth_cfg: dict = config.get("depth_model", {})

        self._capture: "KinectCapture | None" = None
        self._converter: "PointCloudConverter | None" = None
        self._detector = None
        self._calibration: "KinectCalibration | None" = None
        self._fps_counter = None

    def setup(self) -> None:
        """카메라, 포인트 클라우드 변환기, PointPillars 모델 로드."""
        from kinect import KinectCapture, KinectCalibration, PointCloudConverter
        from models.depth import PointPillarsDetector
        from utils.fps_counter import FPSCounter

        logger.info("DepthPipeline setup 시작")

        kinect_cfg = self._cfg.get("kinect", {})
        self._capture = KinectCapture(kinect_cfg)
        self._capture.open()

        self._calibration = KinectCalibration(self._capture.device)
        self._converter = PointCloudConverter(
            self._calibration, self._cfg.get("pointcloud", {})
        )

        model_cfg = self._depth_cfg.get("pointpillars", {})
        self._detector = PointPillarsDetector(model_cfg)
        self._detector.load()

        self._fps_counter = FPSCounter()
        self._last_color_frame = None
        self._last_depth_frame = None

        logger.info("DepthPipeline setup 완료")

    def run_once(self) -> "DetectionResult":
        """프레임 1장을 캡처하고 3D 탐지 결과(+ 2D 투영)를 반환합니다."""
        from models import Detection, DetectionResult

        frame = self._capture.get_frame()
        self._last_color_frame = frame.color
        self._last_depth_frame = frame.depth
        point_cloud = self._converter.convert(frame.depth, frame.color)

        result: "DetectionResult" = self._detector.detect(point_cloud)

        # 3D bbox → 2D 투영
        updated_dets = []
        for det in result.detections:
            if det.bbox_3d is not None:
                try:
                    det.bbox_2d = self._project_3d_to_2d(det.bbox_3d)
                except Exception as exc:
                    logger.debug(f"3D→2D 투영 실패: {exc}")
            updated_dets.append(det)

        self._fps_counter.tick()

        return DetectionResult(
            detections=updated_dets,
            inference_time_ms=result.inference_time_ms,
            frame_id=frame.frame_id,
            timestamp_usec=frame.timestamp_usec,
        )

    def run(self) -> None:
        """파이프라인을 설정하고 스트리밍 루프를 실행합니다."""
        from utils.visualizer import Visualizer
        self.setup()
        vis_cfg = self._cfg.get("postprocess", {}).get("visualizer", {})
        vis = Visualizer(vis_cfg)
        vis.open()
        try:
            for result in self.run_stream():
                fps = self._fps_counter.fps if self._fps_counter else 0.0
                if not vis.render(
                    self._last_color_frame,
                    result,
                    depth=self._last_depth_frame,
                    fps=fps,
                ):
                    break
        finally:
            vis.close()

    def cleanup(self) -> None:
        """shutdown()의 별칭."""
        self.shutdown()

    def run_stream(self) -> Generator["DetectionResult", None, None]:
        """연속 프레임을 탐지하는 제너레이터."""
        while True:
            yield self.run_once()

    def shutdown(self) -> None:
        """카메라 및 모델 리소스 해제."""
        logger.info("DepthPipeline shutdown")
        if self._capture is not None:
            self._capture.close()
        if self._detector is not None:
            try:
                self._detector.unload()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _project_3d_to_2d(self, bbox_3d: np.ndarray) -> np.ndarray:
        """3D bbox (7,) → 2D bbox (4,) xyxy 투영.

        bbox_3d: [cx, cy, cz, l, w, h, yaw] (미터, LiDAR 좌표계)
        8 코너를 핀홀 투영 후 axis-aligned 2D bbox 반환.
        """
        cx, cy, cz, l, w, h, yaw = bbox_3d

        # 8 코너 생성 (LiDAR 좌표계, 벡터화)
        dx = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float32) * l / 2
        dy = np.array([1, -1, -1, 1, 1, -1, -1, 1], dtype=np.float32) * w / 2
        dz = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float32) * h / 2

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rx = cos_y * dx - sin_y * dy
        ry = sin_y * dx + cos_y * dy

        corners_lidar = np.stack(
            [cx + rx, cy + ry, cz + dz], axis=1
        )  # (8, 3)

        # LiDAR → Depth Camera 좌표
        X_cam = -corners_lidar[:, 1]
        Y_cam = -corners_lidar[:, 2]
        Z_cam =  corners_lidar[:, 0]

        valid = Z_cam > 0.0
        if not np.any(valid):
            return np.zeros(4, dtype=np.float32)

        X_cam, Y_cam, Z_cam = X_cam[valid], Y_cam[valid], Z_cam[valid]

        intrinsics = self._calibration.depth_intrinsics
        u = intrinsics.fx * (X_cam / Z_cam) + intrinsics.cx
        v = intrinsics.fy * (Y_cam / Z_cam) + intrinsics.cy

        return np.array(
            [u.min(), v.min(), u.max(), v.max()], dtype=np.float32
        )
