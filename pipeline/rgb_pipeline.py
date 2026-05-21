"""
RGBPipeline: RGB 카메라 단독 탐지 모드
- 활성 모델 목록에서 Detector 인스턴스 생성
- rgb_inference_parallel=True 이면 ThreadPoolExecutor로 병렬 추론
- 결과 합산 후 NMS 적용
- FPSCounter로 FPS 측정
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Generator, List, Optional

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from models import DetectionResult, BaseDetector
    from kinect import KinectCapture


_MODEL_REGISTRY: dict = {
    "yolov11": ("models.rgb", "YOLOv11Detector"),
    "rtdetrv2": ("models.rgb", "RTDETRv2Detector"),
    "grounding_dino": ("models.rgb", "GroundingDINODetector"),
}


def _build_detector(name: str, model_cfg: dict) -> "BaseDetector":
    """모델 이름으로 Detector 인스턴스를 생성합니다."""
    import importlib

    if name not in _MODEL_REGISTRY:
        raise ValueError(f"알 수 없는 RGB 모델: {name}")

    module_path, class_name = _MODEL_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(model_cfg)


class RGBPipeline:
    """RGB 단독 탐지 파이프라인."""

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._rgb_cfg: dict = config.get("rgb_models", {})
        self._fusion_cfg: dict = config.get("fusion", {})
        self._perf_cfg: dict = config.get("performance", {})

        self._capture: "KinectCapture | None" = None
        self._detectors: List["BaseDetector"] = []
        self._fps_counter = None
        self._last_color_frame: Optional[np.ndarray] = None
        self._parallel: bool = self._perf_cfg.get("rgb_inference_parallel", False)
        self._nms_iou: float = self._fusion_cfg.get("iou_threshold", 0.5)
        # ThreadPoolExecutor를 매 프레임마다 재생성하지 않도록 setup()에서 한 번만 생성
        self._executor: Optional[ThreadPoolExecutor] = None

    def setup(self) -> None:
        """모델 로드 및 카메라 오픈."""
        from kinect import KinectCapture
        from utils.fps_counter import FPSCounter

        logger.info("RGBPipeline setup 시작")

        # 카메라 오픈
        self._capture = KinectCapture(self._cfg.get("kinect", {}))
        self._capture.open()

        # 활성 모델 로드
        active_models: List[str] = self._rgb_cfg.get("active", [])

        for model_name in active_models:
            det = _build_detector(model_name, self._rgb_cfg.get(model_name, {}))
            det.load()
            self._detectors.append(det)
            logger.info(f"모델 로드 완료: {model_name}")

        self._fps_counter = FPSCounter()

        # ThreadPoolExecutor를 한 번만 생성하여 매 프레임 재생성 오버헤드 제거
        if self._parallel and len(self._detectors) > 1:
            self._executor = ThreadPoolExecutor(max_workers=len(self._detectors))

        logger.info(f"RGBPipeline setup 완료 — 모델 수: {len(self._detectors)}")

    def run_once(self) -> "DetectionResult":
        """프레임 1장을 캡처하고 탐지 결과를 반환합니다."""
        from models import Detection, DetectionResult
        from fusion.nms import nms_2d

        frame = self._capture.get_frame()
        self._last_color_frame = frame.color
        color_img = frame.color

        results: List["DetectionResult"] = []

        if self._parallel and self._executor is not None:
            futures = {
                self._executor.submit(det.detect, color_img): det
                for det in self._detectors
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.warning(f"병렬 추론 실패: {exc}")
        else:
            for det in self._detectors:
                try:
                    results.append(det.detect(color_img))
                except Exception as exc:
                    logger.warning(f"순차 추론 실패: {exc}")

        # 모든 Detection 수집
        all_dets: List[Detection] = []
        total_ms: float = 0.0
        for r in results:
            all_dets.extend(r.detections)
            total_ms += r.inference_time_ms

        # NMS 적용
        if len(all_dets) > 1:
            boxes = np.stack([d.bbox_2d for d in all_dets], axis=0).astype(np.float32)
            scores = np.array([d.score for d in all_dets], dtype=np.float32)
            keep = nms_2d(boxes, scores, self._nms_iou)
            all_dets = [all_dets[i] for i in keep]

        self._fps_counter.tick()

        return DetectionResult(
            detections=all_dets,
            inference_time_ms=total_ms,
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
                if not vis.render(self._last_color_frame, result, fps=fps):
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
        logger.info("RGBPipeline shutdown")
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        if self._capture is not None:
            self._capture.close()
        for det in self._detectors:
            try:
                det.unload()
            except Exception:
                pass
        self._detectors.clear()
