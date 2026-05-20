from __future__ import annotations

from typing import List, Optional

import numpy as np
from loguru import logger

from ..base import BaseDetector, Detection, DetectionResult

try:
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
    _PCDET_AVAILABLE = True
except ImportError:
    _PCDET_AVAILABLE = False
    logger.warning("[PointPillars] OpenPCDet not installed — detector disabled until available.")


class PointPillarsDetector(BaseDetector):
    """LiDAR/depth-based 3D detector using PointPillars via OpenPCDet."""

    def __init__(self, config: dict, device: str = "cuda:0") -> None:
        super().__init__(config, device)
        self.cfg_file: str = config["cfg_file"]
        self.weights: str = config["weights"]
        self._cfg = None

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        if not _PCDET_AVAILABLE:
            logger.error("[PointPillars] Cannot load model: OpenPCDet is not installed.")
            return

        import torch

        cfg_from_yaml_file(self.cfg_file, cfg)
        self._cfg = cfg

        logger.info(f"[PointPillars] building network from {self.cfg_file}")
        model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=None,
        )

        ckpt = torch.load(self.weights, map_location="cpu")
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict)
        model.eval()
        model.cuda()
        self._model = model
        logger.info("[PointPillars] model loaded")

    # ------------------------------------------------------------------
    def detect(
        self,
        points: np.ndarray,          # (N, 4) [x, y, z, intensity]
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.1,  # NMS IoU (OpenPCDet 내부 처리)
    ) -> DetectionResult:
        if not _PCDET_AVAILABLE:
            raise RuntimeError("OpenPCDet is not installed. Install it to use PointPillarsDetector.")
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch
        import time

        input_dict = {
            "points": torch.from_numpy(points).float(),
            "frame_id": 0,
        }
        data_dict = self._cfg.DATA_CONFIG  # shallow ref for metadata
        batch_dict = {"points": input_dict["points"].unsqueeze(0).cuda()}

        t0 = time.perf_counter()
        with torch.no_grad():
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = self._model.forward(batch_dict)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        detections = self._parse_pred(pred_dicts[0], conf_threshold)
        return DetectionResult(detections=detections, inference_time_ms=inference_ms)

    # ------------------------------------------------------------------
    def _parse_pred(self, pred: dict, conf_threshold: float) -> List[Detection]:
        boxes = pred["pred_boxes"].cpu().numpy()    # (N, 7)
        scores = pred["pred_scores"].cpu().numpy()  # (N,)
        labels = pred["pred_labels"].cpu().numpy().astype(int)  # (N,)

        mask = scores >= conf_threshold
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        class_names = self._cfg.CLASS_NAMES if self._cfg else []
        return [
            Detection(
                bbox_2d=np.zeros(4, dtype=np.float32),  # fusion 단계에서 채워짐
                bbox_3d=boxes[i],
                score=float(scores[i]),
                class_id=int(labels[i]),
                class_name=class_names[int(labels[i]) - 1] if class_names else "",
                source="depth",
            )
            for i in range(len(boxes))
        ]

    # ------------------------------------------------------------------
    @staticmethod
    def project_3d_to_2d(
        bbox_3d: np.ndarray,          # (7,) [cx,cy,cz,l,w,h,yaw]
        calibration: np.ndarray,       # (3,4) 투영 행렬 P
    ) -> np.ndarray:
        """3D bounding box → 2D AABB [x1,y1,x2,y2] (벡터화)."""
        cx, cy, cz, l, w, h, yaw = bbox_3d

        # 8 코너의 로컬 오프셋 (±l/2, ±w/2, ±h/2)
        # shape: (8, 3)
        signs = np.array([
            [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
            [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
        ], dtype=np.float64)
        half_dims = np.array([l / 2, w / 2, h / 2])
        corners_local = signs * half_dims          # (8, 3)

        # yaw 회전 (Z 축 기준)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=np.float64)
        corners_world = corners_local @ R.T + np.array([cx, cy, cz])  # (8, 3)

        # 동차 좌표로 투영
        ones = np.ones((8, 1), dtype=np.float64)
        corners_h = np.hstack([corners_world, ones])   # (8, 4)
        projected = corners_h @ calibration.T           # (8, 3)

        # 깊이 나눗셈
        depth = projected[:, 2:3]
        depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
        pixels = projected[:, :2] / depth              # (8, 2)

        x1, y1 = pixels[:, 0].min(), pixels[:, 1].min()
        x2, y2 = pixels[:, 0].max(), pixels[:, 1].max()
        return np.array([x1, y1, x2, y2], dtype=np.float32)
