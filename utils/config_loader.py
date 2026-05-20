from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml
from loguru import logger


_REQUIRED_KEYS = ("mode", "kinect", "gpu", "rgb_models", "depth_model", "postprocess")
_VALID_MODES = ("rgb", "depth", "fusion")


class ConfigLoader:
    """YAML 설정 파일 로드 및 drone_extension override 처리."""

    def __init__(self, config_path: str) -> None:
        self._config_path = config_path

    def load(self) -> dict:
        """설정 파일을 로드하고 drone_extension override를 적용합니다."""
        with open(self._config_path, "r", encoding="utf-8") as f:
            config: dict = yaml.safe_load(f)

        config = self._apply_drone_extension(config)
        self.validate(config)
        logger.info(f"설정 로드 완료: {self._config_path} (mode={config.get('mode')})")
        return config

    def _apply_drone_extension(self, config: dict) -> dict:
        """drone_extension.enabled=True이면 관련 값을 override합니다."""
        drone_ext: dict = config.get("drone_extension", {})
        if not drone_ext.get("enabled", False):
            return config

        result = deepcopy(config)
        logger.info("drone_extension 활성화: 설정 override 적용 중...")

        yolo_weights = drone_ext.get("yolov11_weights")
        if yolo_weights:
            result.setdefault("rgb_models", {}).setdefault("yolov11", {})["weights"] = yolo_weights
            logger.debug(f"  yolov11.weights → {yolo_weights}")

        gdino_prompt = drone_ext.get("gdino_prompt")
        if gdino_prompt:
            result.setdefault("rgb_models", {}).setdefault("grounding_dino", {})["text_prompt"] = gdino_prompt
            logger.debug(f"  grounding_dino.text_prompt → {gdino_prompt}")

        imgsz = drone_ext.get("imgsz")
        if imgsz:
            for model_key in ("yolov11", "rtdetrv2"):
                result["rgb_models"].setdefault(model_key, {})["imgsz"] = imgsz
            logger.debug(f"  imgsz → {imgsz}")

        min_box_area = drone_ext.get("min_box_area")
        if min_box_area is not None:
            result.setdefault("postprocess", {}).setdefault("tracker", {})["min_box_area"] = min_box_area
            logger.debug(f"  tracker.min_box_area → {min_box_area}")

        conf = drone_ext.get("conf")
        if conf is not None:
            for model_key in ("yolov11", "rtdetrv2"):
                result["rgb_models"].setdefault(model_key, {})["conf"] = conf
            logger.debug(f"  conf → {conf}")

        return result

    @staticmethod
    def validate(config: dict) -> None:
        """필수 키 존재 확인 및 mode 값 검증."""
        missing = [k for k in _REQUIRED_KEYS if k not in config]
        if missing:
            raise ValueError(f"설정 파일에 필수 키가 없습니다: {missing}")

        mode = config.get("mode")
        if mode not in _VALID_MODES:
            raise ValueError(
                f"지원하지 않는 mode: '{mode}'. 허용값: {_VALID_MODES}"
            )
