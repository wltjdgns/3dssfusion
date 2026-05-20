from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

from ..base import BaseDetector, Detection, DetectionResult

try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False
    logger.warning("[YOLOv11] ultralytics not installed — detector disabled until available.")


class YOLOv11Detector(BaseDetector):
    """YOLOv11 RGB detector backed by ultralytics."""

    def __init__(self, config: dict, device: str = "cuda:0") -> None:
        super().__init__(config, device)
        self.weights: str = config["weights"]
        self.imgsz: int = config.get("imgsz", 640)
        self.half: bool = config.get("half", False)
        self.classes: Optional[List[int]] = config.get("classes", None)
        self.augment: bool = config.get("augment", False)
        self.stream: bool = config.get("stream", True)

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        if not _ULTRALYTICS_AVAILABLE:
            logger.error("[YOLOv11] Cannot load model: ultralytics is not installed.")
            return

        logger.info(f"[YOLOv11] loading weights: {self.weights}")
        self._model = _YOLO(self.weights).to(self.device)
        if self.half:
            self._model.half()
        logger.info("[YOLOv11] model loaded")

    # ------------------------------------------------------------------
    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> DetectionResult:
        if not _ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics is not installed. Install it to use YOLOv11Detector.")
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self._model(
            frame,
            imgsz=self.imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            half=self.half,
            classes=self.classes,
            augment=self.augment,
            stream=self.stream,
            verbose=False,
        )

        result = next(iter(results))
        inference_ms: float = result.speed.get("inference", 0.0)
        detections = self._parse_result(result)
        return DetectionResult(detections=detections, inference_time_ms=inference_ms)

    # ------------------------------------------------------------------
    def _parse_result(self, result) -> List[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()        # (N, 4)
        confs = boxes.conf.cpu().numpy()       # (N,)
        cls_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)
        names = result.names                   # dict {id: name}

        return [
            Detection(
                bbox_2d=xyxy[i],
                score=float(confs[i]),
                class_id=int(cls_ids[i]),
                class_name=names.get(int(cls_ids[i]), ""),
                source="rgb",
            )
            for i in range(len(xyxy))
        ]

    # ------------------------------------------------------------------
    def export_onnx(self, output_path: str) -> None:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")
        self._model.export(format="onnx", imgsz=self.imgsz, half=self.half,
                           output=output_path)
        logger.info(f"[YOLOv11] ONNX exported → {output_path}")

    def export_tensorrt(self, output_path: str, fp16: bool = True) -> None:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")
        self._model.export(format="engine", imgsz=self.imgsz, half=fp16,
                           output=output_path)
        logger.info(f"[YOLOv11] TensorRT engine exported → {output_path}")
