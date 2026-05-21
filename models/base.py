from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from loguru import logger


@dataclass
class Detection:
    bbox_2d: np.ndarray           # (4,) [x1, y1, x2, y2] pixel
    bbox_3d: Optional[np.ndarray] = None  # (7,) [cx,cy,cz,l,w,h,yaw] meter
    score: float = 0.0
    class_id: int = 0
    class_name: str = ""
    depth_m: Optional[float] = None
    source: str = "rgb"           # "rgb" | "depth" | "fusion"
    track_id: Optional[int] = None


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    frame_id: int = 0
    timestamp_usec: int = 0


class BaseDetector(ABC):
    def __init__(self, config: dict, device: str = "cuda:0") -> None:
        self.config = config
        self.device = device
        self._model = None
        self._cuda_stream = None

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> DetectionResult: ...

    def load(self) -> None:
        """load_model()의 별칭 — 파이프라인에서 공통 호출 인터페이스로 사용합니다."""
        self.load_model()
        try:
            import torch
            if torch.cuda.is_available():
                self._cuda_stream = torch.cuda.Stream()
        except Exception:
            self._cuda_stream = None

    def unload(self) -> None:
        """모델을 메모리에서 해제합니다."""
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            torch.cuda.empty_cache()

    def warmup(self, n_iter: int = 3) -> None:
        imgsz = self.config.get("imgsz", 640)
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        logger.info(f"[{self.__class__.__name__}] warmup {n_iter} iterations …")
        for _ in range(n_iter):
            self.detect(dummy)
        logger.info(f"[{self.__class__.__name__}] warmup done")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
