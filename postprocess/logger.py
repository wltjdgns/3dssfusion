from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

import numpy as np
from loguru import logger

from models.base import DetectionResult


class DetectionLogger:
    """탐지 결과를 JSONL 또는 CSV 형식으로 기록합니다."""

    _CSV_BUFFER_SIZE = 100

    def __init__(self, config: dict) -> None:
        self._log_dir: str = config.get("log_dir", "data/results/")
        self._log_format: str = config.get("log_format", "jsonl").lower()
        self._log_every_n: int = int(config.get("log_every_n_frames", 1))

        os.makedirs(self._log_dir, exist_ok=True)

        self._file = None
        self._csv_buffer: list[dict] = []

        if self._log_format == "jsonl":
            path = os.path.join(self._log_dir, "detections.jsonl")
            self._file = open(path, "a", encoding="utf-8")
            logger.info(f"DetectionLogger (JSONL): {path}")
        elif self._log_format == "csv":
            self._csv_path = os.path.join(self._log_dir, "detections.csv")
            logger.info(f"DetectionLogger (CSV): {self._csv_path}")
        else:
            raise ValueError(f"지원하지 않는 log_format: '{self._log_format}'. 허용값: jsonl, csv")

    def log(self, result: DetectionResult, frame_id: int) -> None:
        """탐지 결과를 기록합니다. log_every_n_frames에 따라 스킵합니다."""
        if self._log_every_n > 1 and frame_id % self._log_every_n != 0:
            return

        if self._log_format == "jsonl":
            self._write_jsonl(result, frame_id)
        else:
            self._buffer_csv(result, frame_id)

    def flush(self) -> None:
        """버퍼링된 데이터를 디스크에 씁니다."""
        if self._log_format == "jsonl" and self._file:
            self._file.flush()
        elif self._log_format == "csv" and self._csv_buffer:
            self._flush_csv()

    def close(self) -> None:
        """파일 핸들을 닫습니다."""
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None

    def _write_jsonl(self, result: DetectionResult, frame_id: int) -> None:
        record = {
            "timestamp_usec": result.timestamp_usec,
            "frame_id": frame_id,
            "inference_time_ms": result.inference_time_ms,
            "detections": [_serialize_detection(d) for d in result.detections],
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _buffer_csv(self, result: DetectionResult, frame_id: int) -> None:
        for det in result.detections:
            row = _serialize_detection(det)
            row["frame_id"] = frame_id
            row["timestamp_usec"] = result.timestamp_usec
            row["inference_time_ms"] = result.inference_time_ms
            self._csv_buffer.append(row)

        if len(self._csv_buffer) >= self._CSV_BUFFER_SIZE:
            self._flush_csv()

    def _flush_csv(self) -> None:
        if not self._csv_buffer:
            return
        import pandas as pd
        df = pd.DataFrame(self._csv_buffer)
        write_header = not os.path.exists(self._csv_path)
        df.to_csv(self._csv_path, mode="a", header=write_header, index=False, encoding="utf-8")
        self._csv_buffer.clear()


def _serialize_detection(det) -> dict[str, Any]:
    """Detection 객체를 JSON 직렬화 가능한 dict로 변환합니다."""
    return {
        "bbox_2d": det.bbox_2d.tolist() if isinstance(det.bbox_2d, np.ndarray) else list(det.bbox_2d),
        "score": float(det.score),
        "class_id": int(det.class_id),
        "class_name": det.class_name,
        "depth_m": float(det.depth_m) if det.depth_m is not None else None,
        "source": det.source,
        "track_id": int(det.track_id) if det.track_id is not None else None,
    }
