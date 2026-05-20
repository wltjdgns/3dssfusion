from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

try:
    import pyk4a
    from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS, ImageFormat
except ImportError as e:
    raise ImportError(
        "pyk4a를 import할 수 없습니다. "
        "pip install pyk4a==1.4.1 후 Azure Kinect SDK v1.4.1을 "
        "C:\\Program Files\\Azure Kinect SDK v1.4.1\\ 에 설치해 주세요.\n"
        f"원본 오류: {e}"
    )


@dataclass
class CaptureFrame:
    color: np.ndarray        # (H, W, 3) uint8 BGR
    depth: np.ndarray        # (H, W) uint16 mm
    ir: np.ndarray           # (H, W) uint16
    timestamp_usec: int
    device_temp: float


_COLOR_RESOLUTION_MAP: dict[str, ColorResolution] = {
    "720P":  ColorResolution.RES_720P,
    "1080P": ColorResolution.RES_1080P,
    "1440P": ColorResolution.RES_1440P,
    "1536P": ColorResolution.RES_1536P,
    "2160P": ColorResolution.RES_2160P,
    "3072P": ColorResolution.RES_3072P,
}

_DEPTH_MODE_MAP: dict[str, DepthMode] = {
    "NFOV_2X2BINNED": DepthMode.NFOV_2X2BINNED,
    "NFOV_UNBINNED":  DepthMode.NFOV_UNBINNED,
    "WFOV_2X2BINNED": DepthMode.WFOV_2X2BINNED,
    "WFOV_UNBINNED":  DepthMode.WFOV_UNBINNED,
    "PASSIVE_IR":     DepthMode.PASSIVE_IR,
}

_FPS_MAP: dict[int, FPS] = {
    5:  FPS.FPS_5,
    15: FPS.FPS_15,
    30: FPS.FPS_30,
}


def _build_config(config: dict) -> Config:
    color_res_key = str(config.get("color_resolution", "1080P")).upper()
    depth_mode_key = str(config.get("depth_mode", "NFOV_UNBINNED")).upper()
    fps_val = int(config.get("camera_fps", 30))
    sync_only = bool(config.get("synchronized_images_only", True))

    if color_res_key not in _COLOR_RESOLUTION_MAP:
        raise ValueError(
            f"지원하지 않는 color_resolution: '{color_res_key}'. "
            f"허용값: {list(_COLOR_RESOLUTION_MAP.keys())}"
        )
    if depth_mode_key not in _DEPTH_MODE_MAP:
        raise ValueError(
            f"지원하지 않는 depth_mode: '{depth_mode_key}'. "
            f"허용값: {list(_DEPTH_MODE_MAP.keys())}"
        )
    if fps_val not in _FPS_MAP:
        raise ValueError(
            f"지원하지 않는 camera_fps: {fps_val}. 허용값: {list(_FPS_MAP.keys())}"
        )

    return Config(
        color_resolution=_COLOR_RESOLUTION_MAP[color_res_key],
        depth_mode=_DEPTH_MODE_MAP[depth_mode_key],
        camera_fps=_FPS_MAP[fps_val],
        synchronized_images_only=sync_only,
        color_format=ImageFormat.COLOR_BGRA32,
    )


class KinectCapture:
    """pyk4a 기반 Azure Kinect DK 프레임 취득 클래스."""

    def __init__(self, config: dict) -> None:
        self._config_dict = config
        self._device: Optional[PyK4A] = None

    def open(self) -> None:
        k4a_config = _build_config(self._config_dict)
        device_id = int(self._config_dict.get("device_id", 0))

        logger.info(f"Azure Kinect 장치 #{device_id} 연결 시도...")
        try:
            self._device = PyK4A(config=k4a_config, device_id=device_id)
            self._device.start()
        except Exception as e:
            raise ConnectionError(
                f"Azure Kinect 장치를 열 수 없습니다 (device_id={device_id}).\n"
                "확인 사항:\n"
                "  1. USB 3.0 포트에 연결되어 있는지 확인\n"
                "  2. Azure Kinect SDK v1.4.1 설치 위치: "
                "C:\\Program Files\\Azure Kinect SDK v1.4.1\\\n"
                "  3. 다른 프로세스가 장치를 점유하고 있지 않은지 확인\n"
                f"원본 오류: {e}"
            ) from e
        logger.info("Azure Kinect 장치 연결 성공.")

    def close(self) -> None:
        if self._device is not None:
            self._device.stop()
            self._device = None
            logger.info("Azure Kinect 장치 연결 종료.")

    def get_frame(self) -> CaptureFrame:
        """블로킹 방식으로 다음 프레임을 취득합니다."""
        if self._device is None:
            raise RuntimeError("장치가 열려 있지 않습니다. open()을 먼저 호출하세요.")

        capture = self._device.get_capture()

        # BGRA (H,W,4) → BGR (H,W,3): alpha 채널 제거
        bgr = capture.color[..., :3]

        device_temp = 0.0
        if hasattr(self._device, "get_device_temperature"):
            try:
                device_temp = float(self._device.get_device_temperature())
            except Exception:
                pass

        return CaptureFrame(
            color=bgr,
            depth=capture.depth,
            ir=capture.ir,
            timestamp_usec=int(capture.device_timestamp_usec),
            device_temp=device_temp,
        )

    def __enter__(self) -> "KinectCapture":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()
