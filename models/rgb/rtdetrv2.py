from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from PIL import Image

from ..base import BaseDetector, Detection, DetectionResult

_DEFAULT_HF_MODEL = "PekingU/rtdetr_r50vd"

try:
    from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("[RTDETRv2] transformers not installed — detector disabled until available.")


class RTDETRv2Detector(BaseDetector):
    """RT-DETRv2 RGB detector backed by HuggingFace Transformers."""

    def __init__(self, config: dict, device: str = "cuda:0") -> None:
        super().__init__(config, device)
        self.weights: str = config.get("weights", _DEFAULT_HF_MODEL)
        self.imgsz: int = config.get("imgsz", 640)
        self.half: bool = config.get("half", False)
        self.conf_threshold: float = config.get("conf", 0.5)
        self._processor = None

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            logger.error("[RTDETRv2] Cannot load model: transformers is not installed.")
            return

        model_id = self.weights if Path(self.weights).exists() else _DEFAULT_HF_MODEL
        logger.info(f"[RTDETRv2] loading from: {model_id}")

        self._processor = RTDetrImageProcessor.from_pretrained(model_id)
        self._model = RTDetrForObjectDetection.from_pretrained(model_id).to(self.device)
        if self.half:
            self._model = self._model.half()
        self._model.eval()
        logger.info("[RTDETRv2] model loaded")

    # ------------------------------------------------------------------
    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,   # RTDETRv2 내부 처리 (파라미터 일관성 유지)
    ) -> DetectionResult:
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is not installed. Install it to use RTDETRv2Detector.")
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch

        pil_image = Image.fromarray(frame[:, :, ::-1])  # BGR → RGB
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.half:
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self._model(**inputs)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        target_sizes = torch.tensor([[pil_image.height, pil_image.width]]).to(self.device)
        results = self._processor.post_process_object_detection(
            outputs, threshold=conf_threshold, target_sizes=target_sizes
        )

        detections = self._parse_result(results[0])
        return DetectionResult(detections=detections, inference_time_ms=inference_ms)

    # ------------------------------------------------------------------
    def _parse_result(self, result: dict) -> List[Detection]:
        boxes = result["boxes"].cpu().numpy()    # (N, 4) xyxy pixels
        scores = result["scores"].cpu().numpy()  # (N,)
        labels = result["labels"].cpu().numpy().astype(int)  # (N,)
        id2label: dict = self._model.config.id2label

        return [
            Detection(
                bbox_2d=boxes[i],
                score=float(scores[i]),
                class_id=int(labels[i]),
                class_name=id2label.get(int(labels[i]), ""),
                source="rgb",
            )
            for i in range(len(boxes))
        ]

