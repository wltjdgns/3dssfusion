"""
FusionDetector: RGB + Depth Late Fusion
전략: weighted_box_fusion (기본) — 멀티모델 결과를 WBF로 병합하고
      FrustumProjector로 depth_m을 보완합니다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from loguru import logger

from fusion.nms import weighted_box_fusion
from fusion.frustum_projector import FrustumProjector

if TYPE_CHECKING:
    from models import Detection, DetectionResult, BaseDetector


class FusionDetector:
    """RGB 탐지 결과와 Depth 탐지 결과를 Late Fusion으로 통합합니다."""

    def __init__(
        self,
        rgb_detectors: List["BaseDetector"],
        depth_detector: Optional["BaseDetector"],
        projector: FrustumProjector,
        config: dict,
    ) -> None:
        self._rgb_detectors = rgb_detectors
        self._depth_detector = depth_detector
        self._projector = projector
        self._cfg = config

        # 설정 파싱
        self._strategy: str = config.get("strategy", "weighted_box_fusion")
        self._iou_threshold: float = config.get("iou_threshold", 0.5)
        self._depth_weight: float = config.get("depth_weight", 1.0)
        self._depth_confirmation_only: bool = config.get("depth_confirmation_only", False)
        self._depth_augment_rgb: bool = config.get("depth_augment_rgb", True)

        # RGB 모델별 가중치 (config에 없으면 균등 분배)
        self._rgb_weights: List[float] = config.get(
            "rgb_weights",
            [1.0] * len(rgb_detectors),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        rgb_results: List["DetectionResult"],
        depth_result: Optional["DetectionResult"],
        points_3d: Optional[np.ndarray],
        frame_id: int,
        image_shape: Tuple[int, int],
    ) -> "DetectionResult":
        """RGB + Depth 결과를 융합하여 단일 DetectionResult를 반환합니다.

        Args:
            rgb_results: RGB 모델별 DetectionResult 리스트
            depth_result: PointPillars DetectionResult (없으면 None)
            points_3d: (N, 4) 전체 포인트 클라우드 (depth augment 용)
            frame_id: 현재 프레임 번호
            image_shape: (H, W)

        Returns:
            단일 DetectionResult (source="fusion")
        """
        from models import DetectionResult, Detection

        total_inference_ms = sum(r.inference_time_ms for r in rgb_results)
        if depth_result is not None:
            total_inference_ms += depth_result.inference_time_ms

        if self._strategy == "weighted_box_fusion":
            detections = self._fuse_wbf(rgb_results, depth_result, image_shape)
        else:
            # fallback: 단순 concat + NMS
            detections = self._fuse_concat(rgb_results, depth_result)

        # depth augment: frustum으로 depth_m 보완
        if self._depth_augment_rgb and points_3d is not None and len(points_3d) > 0:
            detections = self._depth_augment(detections, points_3d)

        # source 필드 통일
        for det in detections:
            det.source = "fusion"

        return DetectionResult(
            detections=detections,
            inference_time_ms=total_inference_ms,
            frame_id=frame_id,
            timestamp_usec=rgb_results[0].timestamp_usec if rgb_results else 0,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fuse_wbf(
        self,
        rgb_results: List["DetectionResult"],
        depth_result: Optional["DetectionResult"],
        image_shape: Tuple[int, int],
    ) -> List["Detection"]:
        """weighted_box_fusion으로 멀티모델 bbox 병합."""
        from models import Detection

        boxes_list: List[np.ndarray] = []
        scores_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        weights: List[float] = []

        # RGB 모델 결과 수집
        for i, result in enumerate(rgb_results):
            b, s, lb = self._detections_to_arrays(result.detections)
            boxes_list.append(b)
            scores_list.append(s)
            labels_list.append(lb)
            w = self._rgb_weights[i] if i < len(self._rgb_weights) else 1.0
            weights.append(w)

        # Depth 모델 결과 포함 (depth_confirmation_only=False 일 때)
        if depth_result is not None and not self._depth_confirmation_only:
            b, s, lb = self._detections_to_arrays(depth_result.detections)
            boxes_list.append(b)
            scores_list.append(s)
            labels_list.append(lb)
            weights.append(self._depth_weight)

        # 모든 입력이 비어있으면 빈 리스트 반환
        total_boxes = sum(len(b) for b in boxes_list)
        if total_boxes == 0:
            return []

        fused_boxes, fused_scores, fused_labels = weighted_box_fusion(
            boxes_list=boxes_list,
            scores_list=scores_list,
            labels_list=labels_list,
            weights=weights,
            iou_threshold=self._iou_threshold,
            image_shape=image_shape,
        )

        return [
            Detection(
                bbox_2d=fused_boxes[i],
                score=float(fused_scores[i]),
                class_id=int(fused_labels[i]),
                source="fusion",
            )
            for i in range(len(fused_boxes))
        ]

    def _fuse_concat(
        self,
        rgb_results: List["DetectionResult"],
        depth_result: Optional["DetectionResult"],
    ) -> List["Detection"]:
        """단순 concat + NMS fallback."""
        from fusion.nms import nms_2d

        all_dets: List["Detection"] = []
        for result in rgb_results:
            all_dets.extend(result.detections)
        if depth_result is not None:
            all_dets.extend(depth_result.detections)

        if not all_dets:
            return []

        boxes, scores, _ = self._detections_to_arrays(all_dets)
        keep = nms_2d(boxes, scores, self._iou_threshold)
        return [all_dets[i] for i in keep]

    def _depth_augment(
        self,
        detections: List["Detection"],
        points_3d: np.ndarray,
    ) -> List["Detection"]:
        """각 Detection의 depth_m을 Frustum 투영으로 보완합니다."""
        for det in detections:
            try:
                frustum_pts = self._projector.bbox2d_to_frustum_points(
                    det.bbox_2d, points_3d
                )
                stats = self._projector.get_depth_stats(frustum_pts)
                if stats["median_depth"] is not None:
                    det.depth_m = stats["median_depth"]
            except Exception as exc:
                logger.warning(f"depth_augment 실패 (det={det.bbox_2d}): {exc}")
        return detections

    @staticmethod
    def _detections_to_arrays(
        detections: List["Detection"],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detection 리스트를 (boxes, scores, labels) numpy 배열로 변환."""
        if not detections:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        boxes = np.stack([d.bbox_2d for d in detections], axis=0).astype(np.float32)
        scores = np.array([d.score for d in detections], dtype=np.float32)
        labels = np.array([d.class_id for d in detections], dtype=np.float32)
        return boxes, scores, labels
