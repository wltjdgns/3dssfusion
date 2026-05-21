"""
FusionPipeline: RGB + Depth Fusion 핵심 파이프라인
- ThreadPoolExecutor(max_workers=3)로 RGB/Depth 병렬 추론
- FusionDetector.fuse()로 Late Fusion
- skip_depth_every_n: depth 추론 주기적 스킵
- frame_queue_size: Queue 버퍼링
"""
from __future__ import annotations

import importlib
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Generator, List, Optional

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from models import DetectionResult, BaseDetector
    from kinect import KinectCapture, KinectCalibration, PointCloudConverter, CaptureFrame


_MODEL_REGISTRY: dict = {
    "yolov11": ("models.rgb", "YOLOv11Detector"),
    "rtdetrv2": ("models.rgb", "RTDETRv2Detector"),
    "grounding_dino": ("models.rgb", "GroundingDINODetector"),
}


def _build_rgb_detector(name: str, model_cfg: dict) -> "BaseDetector":
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"알 수 없는 RGB 모델: {name}")
    module_path, class_name = _MODEL_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(model_cfg)


class FusionPipeline:
    """RGB + Depth Late Fusion 파이프라인."""

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._rgb_cfg: dict = config.get("rgb_models", {})
        self._depth_cfg: dict = config.get("depth_model", {})
        self._fusion_cfg: dict = config.get("fusion", {})
        self._pipeline_cfg: dict = config.get("performance", {})

        self._capture: "KinectCapture | None" = None
        self._calibration: "KinectCalibration | None" = None
        self._converter: "PointCloudConverter | None" = None
        self._rgb_detectors: List["BaseDetector"] = []
        self._depth_detector = None
        self._fusion_detector = None
        self._fps_counter = None

        self._frame_count: int = 0
        self._last_depth_result: Optional["DetectionResult"] = None
        self._last_points_3d: Optional[np.ndarray] = None
        self._last_color_frame: Optional[np.ndarray] = None
        self._last_depth_frame: Optional[np.ndarray] = None
        self._depth_cache_lock = threading.Lock()  # _last_depth_result 동시 접근 보호

        self._skip_depth_every_n: int = self._pipeline_cfg.get("skip_depth_every_n", 1)
        self._frame_queue_size: int = self._pipeline_cfg.get("frame_queue_size", 4)
        self._depth_async: bool = self._pipeline_cfg.get("depth_inference_async", True)
        self._warmup_frames: int = self._pipeline_cfg.get("warmup_frames", 5)

        self._frame_queue: queue.Queue = queue.Queue(maxsize=self._frame_queue_size)
        # ThreadPoolExecutor를 매 프레임마다 재생성하지 않도록 setup()에서 한 번만 생성
        self._executor: Optional[ThreadPoolExecutor] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """모든 RGB + Depth 모델 로드, KinectCapture 오픈, warmup."""
        from kinect import KinectCapture, KinectCalibration, PointCloudConverter
        from models.depth import PointPillarsDetector
        from fusion.frustum_projector import FrustumProjector
        from fusion.fusion_detector import FusionDetector
        from utils.fps_counter import FPSCounter

        logger.info("FusionPipeline setup 시작")

        # Kinect 초기화 — calibration은 device가 열린 뒤에 생성해야 함
        kinect_cfg = self._cfg.get("kinect", {})
        self._capture = KinectCapture(kinect_cfg)
        self._capture.open()
        self._calibration = KinectCalibration(self._capture.device)
        self._converter = PointCloudConverter(
            self._calibration, self._cfg.get("pointcloud", {})
        )

        # RGB 모델 로드
        active_models: List[str] = self._rgb_cfg.get("active", [])
        for name in active_models:
            det = _build_rgb_detector(name, self._rgb_cfg.get(name, {}))
            det.load()
            self._rgb_detectors.append(det)
            logger.info(f"RGB 모델 로드: {name}")

        # Depth 모델 로드
        pp_cfg = self._depth_cfg.get("pointpillars", {})
        self._depth_detector = PointPillarsDetector(pp_cfg)
        self._depth_detector.load()
        logger.info("PointPillars 모델 로드 완료")

        # Fusion 구성
        projector = FrustumProjector(self._calibration)
        self._fusion_detector = FusionDetector(
            rgb_detectors=self._rgb_detectors,
            depth_detector=self._depth_detector,
            projector=projector,
            config=self._fusion_cfg,
        )

        self._fps_counter = FPSCounter()

        # ThreadPoolExecutor를 한 번만 생성하여 매 프레임 재생성 오버헤드 제거
        # RGB 모델 수 + Depth 1 (PC변환+추론 단일 task로 실행)
        n_workers = len(self._rgb_detectors) + 1
        self._executor = ThreadPoolExecutor(max_workers=max(n_workers, 3))

        # Warmup
        self._run_warmup()
        logger.info("FusionPipeline setup 완료")

    def run(self) -> None:
        """파이프라인을 설정하고 스트리밍 루프를 실행합니다."""
        from utils.visualizer import Visualizer
        self.setup()
        vis_cfg = self._cfg.get("postprocess", {}).get("visualizer", {})
        vis = Visualizer(vis_cfg)
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

    def run_once(self) -> "DetectionResult":
        """프레임 1장을 처리하고 Fusion 결과를 반환합니다."""
        from models import DetectionResult

        frame = self._capture.get_frame()
        self._frame_count += 1
        self._last_color_frame = frame.color
        self._last_depth_frame = frame.depth
        image_shape = (frame.color.shape[0], frame.color.shape[1])

        # Depth 스킵 여부 판단
        run_depth = (self._frame_count % self._skip_depth_every_n == 0)

        rgb_results: List["DetectionResult"] = []
        depth_result: Optional["DetectionResult"] = None
        points_3d: Optional[np.ndarray] = None

        # 영속 ThreadPoolExecutor로 RGB / Depth / PC변환 병렬 실행
        executor = self._executor
        rgb_futures: List[Future] = [
            executor.submit(det.detect, frame.color)
            for det in self._rgb_detectors
        ]

        # Depth 추론 (스킵 안 할 때만): PC변환도 thread pool에서 실행해 main thread 블록 해제
        depth_future: Optional[Future] = None
        _pc_ref: list = []  # 클로저로 points_3d 전달

        if run_depth and self._depth_detector is not None:
            depth_frame = frame.depth
            color_frame = frame.color
            converter = self._converter
            detector = self._depth_detector

            def _depth_task():
                pc = converter.convert(depth_frame, color_frame)
                _pc_ref.append(pc)
                return detector.detect(pc)

            depth_future = executor.submit(_depth_task)

        # RGB 결과 수집
        for future in rgb_futures:
            try:
                rgb_results.append(future.result())
            except Exception as exc:
                logger.warning(f"RGB 추론 실패: {exc}")

        # Depth 결과 수집
        if depth_future is not None:
            try:
                depth_result = depth_future.result()
                points_3d = _pc_ref[0] if _pc_ref else None
                with self._depth_cache_lock:
                    self._last_depth_result = depth_result
                    self._last_points_3d = points_3d
            except Exception as exc:
                logger.warning(f"Depth 추론 실패: {exc}")
                with self._depth_cache_lock:
                    depth_result = self._last_depth_result
                    points_3d = self._last_points_3d
        else:
            # 이전 depth 결과 재사용 (thread-safe read)
            with self._depth_cache_lock:
                depth_result = self._last_depth_result
                points_3d = self._last_points_3d

        # Fusion
        fused = self._fusion_detector.fuse(
            rgb_results=rgb_results,
            depth_result=depth_result,
            points_3d=points_3d,
            frame_id=frame.frame_id,
            image_shape=image_shape,
        )

        self._fps_counter.tick()

        # Queue 버퍼 (비블로킹)
        try:
            self._frame_queue.put_nowait(fused)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(fused)
            except queue.Empty:
                pass

        return fused

    def run_stream(self) -> Generator["DetectionResult", None, None]:
        """연속 프레임을 처리하는 제너레이터."""
        while True:
            yield self.run_once()

    def shutdown(self) -> None:
        """모든 리소스 해제."""
        logger.info("FusionPipeline shutdown")
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        if self._capture is not None:
            self._capture.close()
        for det in self._rgb_detectors:
            try:
                det.unload()
            except Exception:
                pass
        if self._depth_detector is not None:
            try:
                self._depth_detector.unload()
            except Exception:
                pass
        self._rgb_detectors.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_warmup(self) -> None:
        """Warmup 프레임으로 모델 예열."""
        logger.info(f"Warmup 시작 ({self._warmup_frames} 프레임)")
        dummy_shape = (1080, 1920, 3)
        dummy_img = np.zeros(dummy_shape, dtype=np.uint8)

        for i in range(self._warmup_frames):
            for det in self._rgb_detectors:
                try:
                    det.detect(dummy_img)
                except Exception:
                    pass
        logger.info("Warmup 완료")
