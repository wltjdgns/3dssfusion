from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

try:
    import pyk4a
    from pyk4a import PyK4A, CalibrationType
except ImportError as e:
    raise ImportError(
        "pyk4a를 import할 수 없습니다. pip install pyk4a==1.4.1 을 실행하세요.\n"
        f"원본 오류: {e}"
    ) from e


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist_coeffs: np.ndarray  # (8,) [k1,k2,p1,p2,k3,k4,k5,k6]


@dataclass
class ExtrinsicTransform:
    rotation: np.ndarray     # (3, 3)
    translation: np.ndarray  # (3,) mm


def _extract_intrinsics(raw_cam_calib) -> CameraIntrinsics:
    """pyk4a raw calibration 구조체에서 CameraIntrinsics를 추출합니다."""
    params = raw_cam_calib.intrinsics.parameters.param
    return CameraIntrinsics(
        fx=float(params.fx),
        fy=float(params.fy),
        cx=float(params.cx),
        cy=float(params.cy),
        width=int(raw_cam_calib.resolution_width),
        height=int(raw_cam_calib.resolution_height),
        dist_coeffs=np.array(
            [params.k1, params.k2, params.p1, params.p2,
             params.k3, params.k4, params.k5, params.k6],
            dtype=np.float64,
        ),
    )


def _extract_extrinsics(raw_calib, from_type: CalibrationType, to_type: CalibrationType) -> ExtrinsicTransform:
    """pyk4a raw calibration 구조체에서 from→to 외부 파라미터를 추출합니다."""
    ext = raw_calib.extrinsics[from_type.value][to_type.value]
    R = np.array(ext.rotation, dtype=np.float64).reshape(3, 3)
    t = np.array(ext.translation, dtype=np.float64)  # mm
    return ExtrinsicTransform(rotation=R, translation=t)


class KinectCalibration:
    """Azure Kinect 캘리브레이션 파라미터 관리 및 좌표 변환."""

    def __init__(self, device: PyK4A) -> None:
        self._raw = device.calibration._calibration
        self._color_intrinsics: Optional[CameraIntrinsics] = None
        self._depth_intrinsics: Optional[CameraIntrinsics] = None
        self._depth_to_color_ext: Optional[ExtrinsicTransform] = None
        logger.debug("KinectCalibration 초기화 완료.")

    # ------------------------------------------------------------------
    # Properties (lazy extraction)
    # ------------------------------------------------------------------

    @property
    def color_intrinsics(self) -> CameraIntrinsics:
        if self._color_intrinsics is None:
            self._color_intrinsics = _extract_intrinsics(
                self._raw.color_camera_calibration
            )
        return self._color_intrinsics

    @property
    def depth_intrinsics(self) -> CameraIntrinsics:
        if self._depth_intrinsics is None:
            self._depth_intrinsics = _extract_intrinsics(
                self._raw.depth_camera_calibration
            )
        return self._depth_intrinsics

    @property
    def depth_to_color_extrinsics(self) -> ExtrinsicTransform:
        if self._depth_to_color_ext is None:
            self._depth_to_color_ext = _extract_extrinsics(
                self._raw,
                from_type=CalibrationType.DEPTH,
                to_type=CalibrationType.COLOR,
            )
        return self._depth_to_color_ext

    # ------------------------------------------------------------------
    # 좌표 변환 (벡터화)
    # ------------------------------------------------------------------

    @staticmethod
    def _undistort_points(
        x_n: np.ndarray, y_n: np.ndarray, dist: np.ndarray, iters: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """왜곡된 정규화 좌표를 왜곡 보정합니다 (벡터화, OpenCV 호환).

        Rational+Brown-Conrady 모델:
            k = [k1, k2, p1, p2, k3, k4, k5, k6]
            r² = x²+y²
            radial_num   = 1 + k1·r² + k2·r⁴ + k3·r⁶
            radial_denom = 1 + k4·r² + k5·r⁴ + k6·r⁶
            x_dist = x·(num/denom) + 2·p1·xy + p2·(r²+2x²)
            → 역산은 반복(fixed-point iteration)으로 수행
        """
        k1, k2, p1, p2, k3, k4, k5, k6 = dist

        x, y = x_n.copy(), y_n.copy()
        for _ in range(iters):
            r2 = x * x + y * y
            r4, r6 = r2 * r2, r2 * r2 * r2
            num   = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            denom = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
            ratio = num / np.where(denom != 0, denom, np.finfo(np.float64).eps)
            x_corr = (x_n - 2.0 * p1 * x * y - p2 * (r2 + 2.0 * x * x)) / ratio
            y_corr = (y_n - p1 * (r2 + 2.0 * y * y) - 2.0 * p2 * x * y) / ratio
            x, y = x_corr, y_corr
        return x, y

    def depth_pixel_to_3d(
        self, depth_pixel: np.ndarray, depth_values: np.ndarray
    ) -> np.ndarray:
        """Depth 픽셀 좌표를 3D XYZ(m)로 역투영합니다 (왜곡 보정 포함).

        파이프라인:
            1. 픽셀 → 왜곡된 정규화 좌표: x_d = (u - cx) / fx
            2. 반복 왜곡 보정 → 이상적 정규화 좌표 (x, y)
            3. X = x * Z,  Y = y * Z,  Z = depth_mm / 1000.0

        Args:
            depth_pixel: (N, 2) [u, v] float or int
            depth_values: (N,) depth in mm (uint16 or float)

        Returns:
            xyz: (N, 3) XYZ in meters, Kinect depth cam 좌표계
        """
        intr = self.depth_intrinsics
        Z = depth_values.astype(np.float64) / 1000.0  # mm → m

        # 왜곡된 정규화 좌표
        x_d = (depth_pixel[:, 0].astype(np.float64) - intr.cx) / intr.fx
        y_d = (depth_pixel[:, 1].astype(np.float64) - intr.cy) / intr.fy

        # 왜곡 보정 (Rational+Brown-Conrady 역산, 5회 반복)
        x, y = self._undistort_points(x_d, y_d, intr.dist_coeffs)

        X = x * Z
        Y = y * Z
        return np.stack([X, Y, Z], axis=1).astype(np.float64)

    def depth_pixel_to_color_pixel(
        self, depth_pixel: np.ndarray, depth_values: np.ndarray
    ) -> np.ndarray:
        """Depth 픽셀 좌표를 Color 픽셀 좌표로 변환합니다.

        변환 파이프라인 (벡터화):
            1. Depth 카메라 역투영 (왜곡 보정 포함) → 3D XYZ (depth cam 좌표계)
            2. Depth→Color 외부 파라미터 적용 → 3D XYZ (color cam 좌표계)
            3. Color 카메라 왜곡 적용 후 핀홀 투영 → 2D (u, v)

        Args:
            depth_pixel: (N, 2) [u, v]
            depth_values: (N,) depth in mm

        Returns:
            color_pixel: (N, 2) [u, v] in color image
        """
        # Step 1: Depth cam 역투영 (왜곡 보정 포함)
        xyz_depth = self.depth_pixel_to_3d(depth_pixel, depth_values)  # (N,3) m

        # Step 2: 외부 파라미터 변환 (mm → m 후 R, t 적용)
        ext = self.depth_to_color_extrinsics
        t_m = ext.translation / 1000.0  # mm → m
        xyz_color = (ext.rotation @ xyz_depth.T).T + t_m  # (N,3)

        # Step 3: Color 카메라 왜곡 적용 후 투영
        intr = self.color_intrinsics
        Z_c = xyz_color[:, 2]
        safe_Z = np.where(Z_c > 0, Z_c, np.finfo(np.float64).eps)

        # 이상적 정규화 좌표
        x_n = xyz_color[:, 0] / safe_Z
        y_n = xyz_color[:, 1] / safe_Z

        # Brown-Conrady 왜곡 적용 (forward distortion: 이상 → 왜곡)
        k1, k2, p1, p2, k3, k4, k5, k6 = intr.dist_coeffs
        r2 = x_n * x_n + y_n * y_n
        r4, r6 = r2 * r2, r2 * r2 * r2
        num   = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        denom = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        ratio = num / np.where(denom != 0, denom, np.finfo(np.float64).eps)
        x_dist = x_n * ratio + 2.0 * p1 * x_n * y_n + p2 * (r2 + 2.0 * x_n * x_n)
        y_dist = y_n * ratio + p1 * (r2 + 2.0 * y_n * y_n) + 2.0 * p2 * x_n * y_n

        u_c = intr.fx * x_dist + intr.cx
        v_c = intr.fy * y_dist + intr.cy
        return np.stack([u_c, v_c], axis=1)

    # ------------------------------------------------------------------
    # 저장 / 불러오기 (JSON)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """캘리브레이션 파라미터를 JSON 파일로 저장합니다."""
        ext = self.depth_to_color_extrinsics
        data = {
            "color_intrinsics": {
                "fx": self.color_intrinsics.fx,
                "fy": self.color_intrinsics.fy,
                "cx": self.color_intrinsics.cx,
                "cy": self.color_intrinsics.cy,
                "width": self.color_intrinsics.width,
                "height": self.color_intrinsics.height,
                "dist_coeffs": self.color_intrinsics.dist_coeffs.tolist(),
            },
            "depth_intrinsics": {
                "fx": self.depth_intrinsics.fx,
                "fy": self.depth_intrinsics.fy,
                "cx": self.depth_intrinsics.cx,
                "cy": self.depth_intrinsics.cy,
                "width": self.depth_intrinsics.width,
                "height": self.depth_intrinsics.height,
                "dist_coeffs": self.depth_intrinsics.dist_coeffs.tolist(),
            },
            "depth_to_color_extrinsics": {
                "rotation": ext.rotation.tolist(),
                "translation": ext.translation.tolist(),
            },
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"캘리브레이션 저장 완료: {path}")

    @classmethod
    def load(cls, path: str) -> "KinectCalibration":
        """JSON 파일에서 캘리브레이션 파라미터를 불러옵니다.

        Returns:
            device 없이 파라미터만 보유하는 KinectCalibration 인스턴스.
        """

        def _parse_intrinsics(d: dict) -> CameraIntrinsics:
            return CameraIntrinsics(
                fx=float(d["fx"]),
                fy=float(d["fy"]),
                cx=float(d["cx"]),
                cy=float(d["cy"]),
                width=int(d["width"]),
                height=int(d["height"]),
                dist_coeffs=np.array(d["dist_coeffs"], dtype=np.float64),
            )

        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"캘리브레이션 파일 로드 실패: {path}\n원본 오류: {e}"
            ) from e

        obj = object.__new__(cls)
        obj._raw = None
        obj._color_intrinsics = _parse_intrinsics(data["color_intrinsics"])
        obj._depth_intrinsics = _parse_intrinsics(data["depth_intrinsics"])
        ext_data = data["depth_to_color_extrinsics"]
        obj._depth_to_color_ext = ExtrinsicTransform(
            rotation=np.array(ext_data["rotation"], dtype=np.float64),
            translation=np.array(ext_data["translation"], dtype=np.float64),
        )
        logger.info(f"캘리브레이션 로드 완료: {path}")
        return obj
