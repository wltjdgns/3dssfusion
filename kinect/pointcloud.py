from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from .calibration import KinectCalibration


class PointCloudConverter:
    """Depth 이미지를 PointPillars 입력용 Point Cloud로 변환합니다.

    좌표계 변환 (Kinect depth cam → LiDAR 표준):
        Kinect: X→right, Y→down, Z→forward
        LiDAR:  X→forward, Y→left, Z→up
        변환식: [X_l, Y_l, Z_l] = [Z_k, -X_k, -Y_k]
    """

    def __init__(self, calibration: "KinectCalibration", config: dict) -> None:
        self._calib = calibration
        self._min_range_m: float = float(config.get("min_range_m", 0.3))
        self._max_range_m: float = float(config.get("max_range_m", 5.0))
        self._remove_ground: bool = bool(config.get("remove_ground", True))
        self._ground_z_thresh: float = float(config.get("ground_z_threshold", -1.5))
        self._max_points: Optional[int] = config.get("max_points", None)

    def depth_to_pointcloud(
        self,
        depth: np.ndarray,
        color: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Depth 이미지 (H,W) uint16 mm → Point Cloud (N,4) float32 [x,y,z,intensity].

        모든 연산은 NumPy 벡터화로 처리합니다.
        """
        H, W = depth.shape

        # (H,W) 전체 픽셀 좌표 생성 → (H*W, 2) [u, v]
        u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
        pixels = np.stack([u_coords.ravel(), v_coords.ravel()], axis=1)  # (H*W, 2)
        depth_flat = depth.ravel().astype(np.float32)                     # (H*W,)

        # 유효 depth 마스크 (0mm 제외)
        valid_mask = depth_flat > 0
        pixels_valid = pixels[valid_mask]       # (M, 2)
        depths_valid = depth_flat[valid_mask]   # (M,)

        if pixels_valid.shape[0] == 0:
            logger.warning("유효한 depth 픽셀이 없습니다.")
            return np.zeros((0, 4), dtype=np.float32)

        # Kinect 좌표계 XYZ (mm) → 역투영
        xyz_kinect = self._calib.depth_pixel_to_3d(pixels_valid, depths_valid)  # (M,3) m

        # LiDAR 좌표계 변환: [Z_k, -X_k, -Y_k]
        xyz_lidar = np.stack(
            [xyz_kinect[:, 2], -xyz_kinect[:, 0], -xyz_kinect[:, 1]],
            axis=1,
        )  # (M,3)

        # 유효 범위 필터: LiDAR X축 (= Kinect Z = depth 방향) 기준
        range_mask = (
            (xyz_lidar[:, 0] >= self._min_range_m) &
            (xyz_lidar[:, 0] <= self._max_range_m)
        )
        xyz_lidar = xyz_lidar[range_mask]

        # 지면 제거: LiDAR Z축 (= -Y_kinect = 상방향) 기준
        if self._remove_ground:
            ground_mask = xyz_lidar[:, 2] >= self._ground_z_thresh
            xyz_lidar = xyz_lidar[ground_mask]

        N = xyz_lidar.shape[0]
        if N == 0:
            return np.zeros((0, 4), dtype=np.float32)

        # 포인트 수 제한: 랜덤 다운샘플링
        if self._max_points is not None and N > self._max_points:
            indices = np.random.choice(N, size=self._max_points, replace=False)
            xyz_lidar = xyz_lidar[indices]

        # intensity = 1.0 (Kinect는 intensity 정보 없음)
        intensity = np.ones((xyz_lidar.shape[0], 1), dtype=np.float32)
        points = np.concatenate([xyz_lidar.astype(np.float32), intensity], axis=1)  # (N,4)

        logger.debug(f"Point Cloud 생성 완료: {points.shape[0]}개 포인트")
        return points

    def to_openpcdet_input(self, points: np.ndarray) -> dict:
        """(N,4) Point Cloud → OpenPCDet DataDict 형식."""
        return {
            "points": points,
            "frame_id": 0,
            "use_lead_xyz": True,
        }
