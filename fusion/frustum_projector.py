"""
FrustumProjector: RGB 2D bbox 영역에 해당하는 포인트 추출 및 깊이 통계
좌표계 변환: LiDAR → Kinect Depth Camera
  X_cam = -Y_lidar
  Y_cam = -Z_lidar
  Z_cam =  X_lidar  (Z_cam > 0 이 카메라 전방)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from kinect import KinectCalibration


class FrustumProjector:
    """LiDAR 포인트를 depth 카메라 이미지 평면에 투영하여 frustum 내 포인트를 추출합니다."""

    def __init__(self, calibration: "KinectCalibration") -> None:
        self._calib = calibration

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bbox2d_to_frustum_points(
        self,
        bbox_2d: np.ndarray,
        points_3d: np.ndarray,
        margin_px: int = 5,
    ) -> np.ndarray:
        """bbox_2d 영역에 투영되는 LiDAR 포인트를 반환합니다.

        Args:
            bbox_2d: (4,) [x1, y1, x2, y2] 픽셀 좌표
            points_3d: (N, 4) LiDAR 좌표계 포인트 [X, Y, Z, intensity]
            margin_px: bbox 확장 마진 (픽셀)

        Returns:
            (M, 4) frustum 내 포인트 (LiDAR 원본 좌표 유지)
        """
        if points_3d.shape[0] == 0:
            return np.zeros((0, 4), dtype=points_3d.dtype)

        # 1. LiDAR → Depth Camera 좌표 변환 (벡터화)
        #    [X_kinect, Y_kinect, Z_kinect] = [-Y_lidar, -Z_lidar, X_lidar]
        X_cam = -points_3d[:, 1]   # -Y_lidar
        Y_cam = -points_3d[:, 2]   # -Z_lidar
        Z_cam =  points_3d[:, 0]   #  X_lidar

        # 2. Z > 0 (카메라 전방) 필터링 — 이후 나눗셈 안전
        valid_z = Z_cam > 0.0
        if not np.any(valid_z):
            return np.zeros((0, 4), dtype=points_3d.dtype)

        X_cam = X_cam[valid_z]
        Y_cam = Y_cam[valid_z]
        Z_cam = Z_cam[valid_z]
        pts_valid = points_3d[valid_z]

        # 3. 핀홀 투영
        intrinsics = self._calib.depth_intrinsics
        fx = intrinsics.fx
        fy = intrinsics.fy
        cx = intrinsics.cx
        cy = intrinsics.cy

        u = fx * (X_cam / Z_cam) + cx   # (M,)
        v = fy * (Y_cam / Z_cam) + cy   # (M,)

        # 4. bbox + margin 마스크
        x1, y1, x2, y2 = bbox_2d
        in_bbox = (
            (u >= x1 - margin_px) &
            (u <= x2 + margin_px) &
            (v >= y1 - margin_px) &
            (v <= y2 + margin_px)
        )

        return pts_valid[in_bbox]

    def get_depth_stats(
        self,
        frustum_points: np.ndarray,
    ) -> dict:
        """frustum 포인트의 깊이(Z_cam = X_lidar) 통계를 반환합니다.

        Args:
            frustum_points: (M, 4) LiDAR 좌표계 포인트

        Returns:
            {'mean_depth', 'median_depth', 'min_depth', 'point_count'}
            포인트가 없으면 깊이 값들은 None
        """
        if frustum_points.shape[0] == 0:
            return {
                "mean_depth": None,
                "median_depth": None,
                "min_depth": None,
                "point_count": 0,
            }

        # Z_cam = X_lidar (전방 거리)
        depths = frustum_points[:, 0]

        return {
            "mean_depth":   float(np.mean(depths)),
            "median_depth": float(np.median(depths)),
            "min_depth":    float(np.min(depths)),
            "point_count":  int(depths.shape[0]),
        }
