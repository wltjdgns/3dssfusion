from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from loguru import logger
from PIL import Image

from ..base import BaseDetector, Detection, DetectionResult

try:
    from groundingdino.util.inference import load_model as _gd_load_model, predict as _gd_predict
    _GDINO_AVAILABLE = True
except ImportError:
    _GDINO_AVAILABLE = False
    logger.warning("[GroundingDINO] groundingdino not installed — detector disabled until available.")


class GroundingDINODetector(BaseDetector):
    """Open-vocabulary detector using GroundingDINO."""

    def __init__(self, config: dict, device: str = "cuda:0") -> None:
        super().__init__(config, device)
        self.config_file: str = config["config_file"]
        self.weights: str = config["weights"]
        self.text_prompt: str = config.get("text_prompt", "person . car . truck .")
        self.box_threshold: float = config.get("box_threshold", 0.35)
        self.text_threshold: float = config.get("text_threshold", 0.25)
        # half=False is fixed for GroundingDINO
        self.half: bool = False

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        if not _GDINO_AVAILABLE:
            logger.error("[GroundingDINO] Cannot load model: groundingdino is not installed.")
            return

        logger.info(f"[GroundingDINO] loading config={self.config_file}, weights={self.weights}")
        self._model = _gd_load_model(self.config_file, self.weights).to(self.device)
        self._model.eval()
        logger.info("[GroundingDINO] model loaded")

    # ------------------------------------------------------------------
    def detect(
        self,
        frame: np.ndarray,
        text_prompt: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ) -> DetectionResult:
        if not _GDINO_AVAILABLE:
            raise RuntimeError("groundingdino is not installed. Install it to use GroundingDINODetector.")
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch
        import torchvision.transforms as T

        prompt = text_prompt if text_prompt is not None else self.text_prompt
        H, W = frame.shape[:2]

        # BGR → RGB PIL → tensor 전처리
        pil_image = Image.fromarray(frame[:, :, ::-1])
        transform = T.Compose([
            T.Resize((800, 1333)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = transform(pil_image).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            boxes, logits, phrases = _gd_predict(
                model=self._model,
                image=image_tensor,
                caption=prompt,
                box_threshold=conf_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
        inference_ms = (time.perf_counter() - t0) * 1000.0

        detections = self._parse_result(boxes, logits, phrases, H, W)
        return DetectionResult(detections=detections, inference_time_ms=inference_ms)

    # ------------------------------------------------------------------
    def _parse_result(
        self,
        boxes: "torch.Tensor",   # (N, 4) cx cy w h normalized
        logits: "torch.Tensor",  # (N,)
        phrases: List[str],
        H: int,
        W: int,
    ) -> List[Detection]:
        if len(boxes) == 0:
            return []

        boxes_np = boxes.cpu().numpy()    # (N, 4)
        scores_np = logits.cpu().numpy()  # (N,)

        # cx,cy,w,h → x1,y1,x2,y2 (벡터화)
        cx, cy, bw, bh = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
        x1 = np.clip((cx - bw / 2) * W, 0, W)
        y1 = np.clip((cy - bh / 2) * H, 0, H)
        x2 = np.clip((cx + bw / 2) * W, 0, W)
        y2 = np.clip((cy + bh / 2) * H, 0, H)
        xyxy = np.stack([x1, y1, x2, y2], axis=1)  # (N, 4)

        return [
            Detection(
                bbox_2d=xyxy[i],
                score=float(scores_np[i]),
                class_name=phrases[i],
                source="rgb",
            )
            for i in range(len(xyxy))
        ]

    # ------------------------------------------------------------------
    def set_prompt(self, text_prompt: str) -> None:
        """텍스트 프롬프트를 런타임에 교체합니다 (모델 재로드 불필요)."""
        self.text_prompt = text_prompt
        logger.info(f"[GroundingDINO] prompt updated → '{text_prompt}'")
