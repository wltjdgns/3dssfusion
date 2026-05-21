from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from loguru import logger

from ..base import BaseDetector, Detection, DetectionResult

try:
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    _PCDET_AVAILABLE = True
except ImportError:
    _PCDET_AVAILABLE = False
    logger.warning("[PointPillars] OpenPCDet not installed — detector disabled until available.")


def _voxelize(
    points: np.ndarray,
    point_cloud_range: list,
    voxel_size: np.ndarray,
    max_points_per_voxel: int = 32,
    max_voxels: int = 16000,
):
    """순수 NumPy 복셀라이저 — spconv 없이 동작."""
    pcr = np.array(point_cloud_range, dtype=np.float32)
    vs  = voxel_size.astype(np.float32)

    mask = (
        (points[:, 0] >= pcr[0]) & (points[:, 0] < pcr[3]) &
        (points[:, 1] >= pcr[1]) & (points[:, 1] < pcr[4]) &
        (points[:, 2] >= pcr[2]) & (points[:, 2] < pcr[5])
    )
    pts = points[mask]
    if len(pts) == 0:
        C = points.shape[1]
        return (np.zeros((0, max_points_per_voxel, C), np.float32),
                np.zeros((0, 4), np.int32),
                np.zeros((0,), np.int32))

    coords = np.floor((pts[:, :3] - pcr[:3]) / vs).astype(np.int32)
    grid   = np.round((pcr[3:6] - pcr[:3]) / vs).astype(np.int32)
    coords = np.clip(coords, 0, grid - 1)

    keys  = (coords[:, 0] * grid[1] * grid[2]
             + coords[:, 1] * grid[2]
             + coords[:, 2])
    order = np.argsort(keys, kind='stable')
    keys  = keys[order];  pts = pts[order];  coords = coords[order]

    ukeys, idx, counts = np.unique(keys, return_index=True, return_counts=True)
    if len(ukeys) > max_voxels:
        ukeys  = ukeys[:max_voxels]
        idx    = idx[:max_voxels]
        counts = counts[:max_voxels]

    nv = len(ukeys)
    C  = pts.shape[1]
    voxels     = np.zeros((nv, max_points_per_voxel, C), np.float32)
    vox_coords = np.zeros((nv, 4), np.int32)   # [batch_idx, z, y, x]
    num_pts    = np.zeros(nv, np.int32)

    for i, (start, cnt) in enumerate(zip(idx, counts)):
        n = min(cnt, max_points_per_voxel)
        voxels[i, :n] = pts[start:start + n]
        c = coords[start]
        vox_coords[i] = [0, c[2], c[1], c[0]]   # batch_idx=0, z, y, x
        num_pts[i]    = n

    return voxels, vox_coords, num_pts


class PointPillarsDetector(BaseDetector):
    """LiDAR/depth 기반 3D 탐지기 — OpenPCDet PointPillars."""

    def __init__(self, config: dict, device: str = "cuda:0") -> None:
        super().__init__(config, device)
        self.cfg_file: str = config["cfg_file"]
        self.weights:  str = config["weights"]
        self._pcdet_cfg   = None
        self._voxel_size: Optional[np.ndarray] = None
        self._pcr:        Optional[list]        = None

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        if not _PCDET_AVAILABLE:
            logger.error("[PointPillars] OpenPCDet not installed.")
            return

        import torch

        cfg_from_yaml_file(self.cfg_file, cfg)
        self._pcdet_cfg = cfg

        _pcr = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
        _vs  = np.array([0.05, 0.05, 0.1], dtype=np.float32)
        for proc in cfg.DATA_CONFIG.get("DATA_PROCESSOR", []):
            if proc.get("NAME") == "transform_points_to_voxels":
                _vs = np.array(proc["VOXEL_SIZE"], dtype=np.float32)
                break
        _gs = np.round((_pcr[3:6] - _pcr[:3]) / _vs).astype(np.int64)

        self._voxel_size = _vs
        self._pcr        = _pcr.tolist()

        class _PFE:
            num_point_features = 4

        class _DatasetProxy:
            class_names           = cfg.CLASS_NAMES
            point_cloud_range     = _pcr
            voxel_size            = _vs
            grid_size             = _gs
            depth_downsample_factor = None
            point_feature_encoder = _PFE()

        logger.info(f"[PointPillars] building network from {self.cfg_file}")
        model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=_DatasetProxy(),
        )

        ckpt       = torch.load(self.weights, map_location="cpu")
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict)
        model.eval().cuda()
        self._model = model
        logger.info("[PointPillars] model loaded")

    # ------------------------------------------------------------------
    def detect(
        self,
        points: np.ndarray,
        conf_threshold: float = 0.3,
        iou_threshold:  float = 0.1,
    ) -> DetectionResult:
        if not _PCDET_AVAILABLE:
            raise RuntimeError("OpenPCDet not installed.")
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch

        if len(points) == 0:
            return DetectionResult(detections=[], inference_time_ms=0.0)

        voxels, vox_coords, num_pts = _voxelize(
            points, self._pcr, self._voxel_size,
            max_points_per_voxel=32, max_voxels=16000,
        )

        if len(voxels) == 0:
            return DetectionResult(detections=[], inference_time_ms=0.0)

        batch_dict = {
            "voxels":           torch.from_numpy(voxels).float().cuda(),
            "voxel_num_points": torch.from_numpy(num_pts).int().cuda(),
            "voxel_coords":     torch.from_numpy(vox_coords).int().cuda(),
            "batch_size":       1,
        }

        t0 = time.perf_counter()
        with torch.no_grad():
            pred_dicts, _ = self._model.forward(batch_dict)
        ms = (time.perf_counter() - t0) * 1000.0

        return DetectionResult(
            detections=self._parse_pred(pred_dicts[0], conf_threshold),
            inference_time_ms=ms,
        )

    # ------------------------------------------------------------------
    def _parse_pred(self, pred: dict, conf_threshold: float) -> List[Detection]:
        boxes  = pred["pred_boxes"].cpu().numpy()
        scores = pred["pred_scores"].cpu().numpy()
        labels = pred["pred_labels"].cpu().numpy().astype(int)

        mask = scores >= conf_threshold
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        class_names = self._pcdet_cfg.CLASS_NAMES if self._pcdet_cfg else []
        return [
            Detection(
                bbox_2d=np.zeros(4, dtype=np.float32),
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
        bbox_3d:     np.ndarray,
        calibration: np.ndarray,
    ) -> np.ndarray:
        cx, cy, cz, l, w, h, yaw = bbox_3d
        signs = np.array([
            [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
            [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
        ], dtype=np.float64)
        corners = signs * np.array([l / 2, w / 2, h / 2])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]], dtype=np.float64)
        corners_w = corners @ R.T + np.array([cx, cy, cz])
        ones = np.ones((8, 1), dtype=np.float64)
        proj = np.hstack([corners_w, ones]) @ calibration.T
        depth = np.where(np.abs(proj[:, 2:3]) < 1e-6, 1e-6, proj[:, 2:3])
        pix = proj[:, :2] / depth
        return np.array([pix[:, 0].min(), pix[:, 1].min(),
                         pix[:, 0].max(), pix[:, 1].max()], dtype=np.float32)
