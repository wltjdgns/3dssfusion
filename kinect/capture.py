from __future__ import annotations

import queue
import threading
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
    frame_id: int = 0


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
        self._frame_id: int = 0
        self._async_queue: Optional[queue.Queue] = None
        self._stop_event: Optional[threading.Event] = None
        self._capture_thread: Optional[threading.Thread] = None

    @property
    def device(self) -> Optional[PyK4A]:
        """열려 있는 PyK4A 장치 객체를 반환합니다 (캘리브레이션 초기화에 사용)."""
        return self._device

    def open(self) -> None:
        k4a_config = _build_config(self._config_dict)
        device_id = int(self._config_dict.get("device_index", self._config_dict.get("device_id", 0)))

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

    def start_async_capture(self) -> None:
        """백그라운드 캡처 스레드를 시작합니다. 처리 루프와 캡처를 겹쳐 GPU idle 제거."""
        self._async_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="kinect-capture"
        )
        self._capture_thread.start()
        logger.info("비동기 캡처 스레드 시작")

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self.get_frame()
                # 큐가 가득 찬 경우 오래된 프레임 버리고 최신 프레임 유지
                if self._async_queue.full():
                    try:
                        self._async_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._async_queue.put_nowait(frame)
            except Exception as exc:
                if not self._stop_event.is_set():
                    logger.warning(f"캡처 스레드 오류: {exc}")

    def get_latest_frame(self, timeout: float = 2.0) -> CaptureFrame:
        """비동기 큐에서 가장 최근 프레임을 반환합니다 (캡처 스레드가 실행 중이어야 함)."""
        if self._async_queue is None:
            return self.get_frame()
        return self._async_queue.get(timeout=timeout)

    def stop_async_capture(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=3.0)
        self._async_queue = None
        self._stop_event = None
        self._capture_thread = None

    def close(self) -> None:
        self.stop_async_capture()
        if self._device is not None:
            self._device.stop()
            self._device = None
            logger.info("Azure Kinect 장치 연결 종료.")

    def get_frame(self) -> CaptureFrame:
        """블로킹 방식으로 다음 프레임을 취득합니다."""
        if self._device is None:
            raise RuntimeError("장치가 열려 있지 않습니다. open()을 먼저 호출하세요.")

        capture = self._device.get_capture()
        self._frame_id += 1

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
            timestamp_usec=int(capture.color_timestamp_usec),
            device_temp=device_temp,
            frame_id=self._frame_id,
        )

    def __enter__(self) -> "KinectCapture":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()
