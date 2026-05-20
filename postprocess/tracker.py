from __future__ import annotations

from typing import Tuple, List

import numpy as np
from loguru import logger

from models.base import Detection, DetectionResult

try:
    from supervision import ByteTrack, Detections as SvDetections
    _SV_AVAILABLE = True
except ImportError:
    _SV_AVAILABLE = False
    logger.warning("supervision 미설치. 추적 없이 실행 (pip install supervision)")


class ByteTracker:
    """supervision ByteTrack 기반 다중 객체 추적."""

    def __init__(self, config: dict):
        self._cfg = config
        self._min_box_area: int = config.get("min_box_area", 100)
        self._tracker = None
        if _SV_AVAILABLE:
            self._tracker = ByteTrack(
                track_activation_threshold=config.get("track_thresh", 0.5),
                lost_track_buffer=config.get("track_buffer", 30),
                minimum_matching_threshold=config.get("match_thresh", 0.8),
            )

    def update(self, result: DetectionResult, frame_shape: Tuple[int, int]) -> DetectionResult:
        """탐지 결과에 track_id를 할당하고 반환합니다.

        Args:
            result: 모델 탐지 결과
            frame_shape: (H, W) 프레임 크기

        Returns:
            track_id가 채워진 DetectionResult
        """
        if not result.detections:
            return result

        dets = result.detections

        # min_box_area 필터 (벡터화)
        boxes = np.array([d.bbox_2d for d in dets], dtype=np.float32)  # (N,4)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep_mask = areas >= self._min_box_area
        dets = [d for d, k in zip(dets, keep_mask) if k]

        if not dets or not _SV_AVAILABLE or self._tracker is None:
            return DetectionResult(
                detections=dets,
                inference_time_ms=result.inference_time_ms,
                frame_id=result.frame_id,
                timestamp_usec=result.timestamp_usec,
            )

        boxes = np.array([d.bbox_2d for d in dets], dtype=np.float32)
        scores = np.array([d.score for d in dets], dtype=np.float32)
        class_ids = np.array([d.class_id for d in dets], dtype=int)

        sv_dets = SvDetections(
            xyxy=boxes,
            confidence=scores,
            class_id=class_ids,
        )

        tracked = self._tracker.update_with_detections(sv_dets)

        # track_id 매핑: supervision은 tracker_id 배열 반환
        tracked_boxes = tracked.xyxy
        tracked_ids = tracked.tracker_id if tracked.tracker_id is not None else np.full(len(tracked), -1)

        # 원래 Detection 순서와 track_id 매칭 (IoU 기반 nearest match)
        updated: List[Detection] = []
        if len(tracked_boxes) > 0:
            iou_matrix = _batch_iou(boxes, tracked_boxes)  # (N_orig, N_tracked)
            matched_idx = iou_matrix.argmax(axis=1)        # (N_orig,)
            max_iou = iou_matrix.max(axis=1)               # (N_orig,)
            for i, det in enumerate(dets):
                tid = int(tracked_ids[matched_idx[i]]) if max_iou[i] > 0.3 else -1
                det.track_id = tid
                updated.append(det)
        else:
            updated = dets

        return DetectionResult(
            detections=updated,
            inference_time_ms=result.inference_time_ms,
            frame_id=result.frame_id,
            timestamp_usec=result.timestamp_usec,
        )

    def reset(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()


def _batch_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) xyxy → (N,M) IoU 행렬. 벡터화."""
    # boxes_a: (N,1,4), boxes_b: (1,M,4)
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter_area
    return np.where(union > 0, inter_area / union, 0.0)
