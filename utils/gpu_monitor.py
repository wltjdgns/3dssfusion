from __future__ import annotations

from loguru import logger

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


class GPUMonitor:
    """GPU 상태 모니터링 (pynvml 우선, 없으면 torch.cuda fallback)."""

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._nvml_handle = None

        if _NVML_AVAILABLE:
            try:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                logger.debug(f"GPUMonitor: pynvml 초기화 (device={device_index})")
            except Exception as e:
                logger.warning(f"pynvml 핸들 취득 실패: {e}")
                self._nvml_handle = None

    def get_stats(self) -> dict:
        """GPU 통계를 반환합니다.

        Returns:
            vram_used_mb, vram_total_mb, utilization_pct, temp_c
        """
        if self._nvml_handle is not None:
            return self._stats_from_nvml()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return self._stats_from_torch()
        return {"vram_used_mb": 0, "vram_total_mb": 0, "utilization_pct": 0, "temp_c": 0}

    def log_stats(self) -> None:
        """현재 GPU 상태를 loguru로 출력합니다."""
        s = self.get_stats()
        logger.info(
            f"GPU[{self._device_index}] "
            f"VRAM: {s['vram_used_mb']}/{s['vram_total_mb']} MB  "
            f"Util: {s['utilization_pct']}%  "
            f"Temp: {s['temp_c']}°C"
        )

    def _stats_from_nvml(self) -> dict:
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
        temp = pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "vram_used_mb": int(mem.used // 1024 // 1024),
            "vram_total_mb": int(mem.total // 1024 // 1024),
            "utilization_pct": int(util.gpu),
            "temp_c": int(temp),
        }

    def _stats_from_torch(self) -> dict:
        used = torch.cuda.memory_allocated(self._device_index)
        total = torch.cuda.get_device_properties(self._device_index).total_memory
        return {
            "vram_used_mb": int(used // 1024 // 1024),
            "vram_total_mb": int(total // 1024 // 1024),
            "utilization_pct": 0,
            "temp_c": 0,
        }
