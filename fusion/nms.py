"""
NMS 유틸리티: 2D/3D bbox NMS 및 WBF(Weighted Box Fusion)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

if TYPE_CHECKING:
    pass


def nms_2d(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """torchvision NMS로 2D bbox 중복 제거.

    Args:
        boxes: (N, 4) xyxy 픽셀 좌표
        scores: (N,) confidence score
        iou_threshold: IoU 임계값

    Returns:
        keep indices (M,)
    """
    import torch
    import torchvision.ops as ops

    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    boxes_t = torch.from_numpy(boxes.astype(np.float32))
    scores_t = torch.from_numpy(scores.astype(np.float32))
    keep = ops.nms(boxes_t, scores_t, iou_threshold)
    return keep.numpy()


def soft_nms_2d(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Gaussian decay Soft-NMS.

    Args:
        boxes: (N, 4) xyxy 픽셀 좌표
        scores: (N,) confidence score
        sigma: Gaussian decay sigma
        score_threshold: 최소 score 임계값

    Returns:
        keep indices (M,) — score_threshold 초과 박스
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    boxes = boxes.astype(np.float64)
    scores = scores.copy().astype(np.float64)
    n = len(boxes)
    indices = np.arange(n)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    for i in range(n):
        # 현재 최고 score index 찾기
        max_idx = i + int(np.argmax(scores[i:]))
        if max_idx != i:
            # 동시에 swap — 순서 의존 없이 tuple 언패킹으로 안전하게 처리
            boxes[i], boxes[max_idx] = boxes[max_idx].copy(), boxes[i].copy()
            scores[i], scores[max_idx] = scores[max_idx], scores[i]
            areas[i], areas[max_idx] = areas[max_idx], areas[i]
            indices[i], indices[max_idx] = indices[max_idx], indices[i]

        # 나머지 박스와 IoU 계산 (벡터화)
        rest = np.arange(i + 1, n)
        if len(rest) == 0:
            break

        ix1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        iy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        ix2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        iy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        inter_w = np.maximum(0.0, ix2 - ix1)
        inter_h = np.maximum(0.0, iy2 - iy1)
        inter = inter_w * inter_h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-8)

        # Gaussian decay
        scores[rest] *= np.exp(-(iou ** 2) / sigma)

    keep = indices[scores > score_threshold]
    return keep.astype(np.int64)


def weighted_box_fusion(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    weights: List[float],
    iou_threshold: float = 0.5,
    image_shape: Tuple[int, int] = (1080, 1920),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ensemble_boxes WBF로 멀티모델 bbox 융합.

    Args:
        boxes_list: 각 모델의 boxes (N_i, 4) xyxy 픽셀
        scores_list: 각 모델의 scores (N_i,)
        labels_list: 각 모델의 labels (N_i,)
        weights: 모델별 가중치
        iou_threshold: WBF IoU 임계값
        image_shape: (H, W) — 정규화에 사용

    Returns:
        fused_boxes (M, 4) xyxy 픽셀, scores (M,), labels (M,)
    """
    from ensemble_boxes import weighted_boxes_fusion

    h, w = image_shape

    # 픽셀 → [0, 1] 정규화
    norm_boxes_list = []
    for boxes in boxes_list:
        if len(boxes) == 0:
            norm_boxes_list.append(np.zeros((0, 4), dtype=np.float32))
            continue
        nb = boxes.astype(np.float32).copy()
        nb[:, [0, 2]] /= w
        nb[:, [1, 3]] /= h
        nb = np.clip(nb, 0.0, 1.0)
        norm_boxes_list.append(nb)

    # ensemble_boxes는 list-of-list 형식 요구
    boxes_ll = [b.tolist() for b in norm_boxes_list]
    scores_ll = [s.tolist() for s in scores_list]
    labels_ll = [lb.tolist() for lb in labels_list]

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_ll,
        scores_ll,
        labels_ll,
        weights=weights,
        iou_thr=iou_threshold,
        skip_box_thr=0.0,
    )

    # [0, 1] → 픽셀 역변환
    fused_boxes = np.array(fused_boxes, dtype=np.float32)
    if len(fused_boxes) > 0:
        fused_boxes[:, [0, 2]] *= w
        fused_boxes[:, [1, 3]] *= h

    return (
        fused_boxes,
        np.array(fused_scores, dtype=np.float32),
        np.array(fused_labels, dtype=np.float32),
    )
