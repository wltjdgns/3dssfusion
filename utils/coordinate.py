from __future__ import annotations

from typing import Tuple

import numpy as np


def kinect_to_lidar(points_kinect: np.ndarray) -> np.ndarray:
    """Kinect 좌표계 → LiDAR 좌표계 변환.

    Kinect: x-right, y-down, z-forward
    LiDAR:  x-forward, y-left, z-up
    변환: [X_l, Y_l, Z_l] = [Z_k, -X_k, -Y_k]

    Args:
        points_kinect: (N, 3) float32

    Returns:
        (N, 3) float32
    """
    return np.stack([
        points_kinect[:, 2],
        -points_kinect[:, 0],
        -points_kinect[:, 1],
    ], axis=-1)


def lidar_to_kinect(points_lidar: np.ndarray) -> np.ndarray:
    """LiDAR 좌표계 → Kinect 좌표계 역변환.

    변환: [X_k, Y_k, Z_k] = [-Y_l, -Z_l, X_l]

    Args:
        points_lidar: (N, 3) float32

    Returns:
        (N, 3) float32
    """
    return np.stack([
        -points_lidar[:, 1],
        -points_lidar[:, 2],
        points_lidar[:, 0],
    ], axis=-1)


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] → [cx, cy, w, h] 변환.

    Args:
        boxes: (N, 4) float32

    Returns:
        (N, 4) float32
    """
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    w  = boxes[:, 2] - boxes[:, 0]
    h  = boxes[:, 3] - boxes[:, 1]
    return np.stack([cx, cy, w, h], axis=-1)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """[cx, cy, w, h] → [x1, y1, x2, y2] 역변환.

    Args:
        boxes: (N, 4) float32

    Returns:
        (N, 4) float32
    """
    half_w = boxes[:, 2] * 0.5
    half_h = boxes[:, 3] * 0.5
    x1 = boxes[:, 0] - half_w
    y1 = boxes[:, 1] - half_h
    x2 = boxes[:, 0] + half_w
    y2 = boxes[:, 1] + half_h
    return np.stack([x1, y1, x2, y2], axis=-1)


def normalize_boxes(boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """픽셀 좌표 bbox → [0, 1] 정규화.

    Args:
        boxes: (N, 4) [x1, y1, x2, y2] 픽셀
        image_shape: (height, width)

    Returns:
        (N, 4) float32 정규화 좌표
    """
    h, w = image_shape
    scale = np.array([w, h, w, h], dtype=np.float32)
    return boxes.astype(np.float32) / scale


def denormalize_boxes(boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """[0, 1] 정규화 좌표 → 픽셀 bbox 역변환.

    Args:
        boxes: (N, 4) float32 정규화 좌표
        image_shape: (height, width)

    Returns:
        (N, 4) float32 픽셀 좌표
    """
    h, w = image_shape
    scale = np.array([w, h, w, h], dtype=np.float32)
    return boxes.astype(np.float32) * scale
