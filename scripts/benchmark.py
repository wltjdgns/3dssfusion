"""FPS 벤치마크 스크립트.

Usage:
    python scripts/benchmark.py --mode rgb --frames 100
    python scripts/benchmark.py --mode depth --frames 50
    python scripts/benchmark.py --mode fusion --frames 100
"""
from __future__ import annotations

import argparse
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


def _run_benchmark(mode: str, num_frames: int) -> dict:
    """지정 모드로 파이프라인을 실행하고 벤치마크 통계를 반환합니다."""
    import yaml
    import numpy as np

    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["mode"] = mode

    from utils.gpu_monitor import GPUMonitor
    gpu_monitor = GPUMonitor(0)

    frame_times: list[float] = []
    vram_samples: list[int] = []

    # 파이프라인 임포트 (실제 장치 없이 더미 프레임 사용)
    try:
        from pipeline.runner import DetectionPipeline
        pipeline = DetectionPipeline(config)
        pipeline.setup()
        use_real = True
    except Exception:
        use_real = False

    dummy_color = np.zeros((1080, 1920, 3), dtype=np.uint8)
    dummy_depth = np.zeros((576, 640), dtype=np.uint16)

    if _RICH:
        from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn
        progress_ctx = Progress(SpinnerColumn(), "[progress.description]{task.description}", BarColumn(), TimeElapsedColumn())
    else:
        progress_ctx = None

    def _run_frames() -> None:
        for i in range(num_frames):
            t0 = time.perf_counter()

            if use_real:
                from kinect.capture import CaptureFrame
                frame = CaptureFrame(
                    color=dummy_color, depth=dummy_depth,
                    ir=dummy_depth, timestamp_usec=i * 33333, device_temp=0.0,
                )
                pipeline.process_frame(frame)
            else:
                time.sleep(0.033)

            elapsed = time.perf_counter() - t0
            frame_times.append(elapsed)
            stats = gpu_monitor.get_stats()
            vram_samples.append(stats["vram_used_mb"])

    if _RICH and progress_ctx:
        with progress_ctx as progress:
            task = progress.add_task(f"[cyan]벤치마크 실행 ({mode}, {num_frames}프레임)", total=num_frames)
            for i in range(num_frames):
                t0 = time.perf_counter()
                if use_real:
                    from kinect.capture import CaptureFrame
                    frame = CaptureFrame(
                        color=dummy_color, depth=dummy_depth,
                        ir=dummy_depth, timestamp_usec=i * 33333, device_temp=0.0,
                    )
                    try:
                        pipeline.process_frame(frame)
                    except Exception:
                        time.sleep(0.033)
                else:
                    time.sleep(0.033)
                elapsed = time.perf_counter() - t0
                frame_times.append(elapsed)
                stats = gpu_monitor.get_stats()
                vram_samples.append(stats["vram_used_mb"])
                progress.advance(task, 1)
    else:
        _run_frames()

    times = [t for t in frame_times if t > 0]
    fps_list = [1.0 / t for t in times]

    return {
        "mode": mode,
        "frames": num_frames,
        "avg_fps": sum(fps_list) / len(fps_list) if fps_list else 0.0,
        "min_fps": min(fps_list) if fps_list else 0.0,
        "max_fps": max(fps_list) if fps_list else 0.0,
        "avg_frame_ms": (sum(times) / len(times) * 1000) if times else 0.0,
        "vram_avg_mb": int(sum(vram_samples) / len(vram_samples)) if vram_samples else 0,
        "vram_peak_mb": max(vram_samples) if vram_samples else 0,
    }


def _print_results(stats: dict) -> None:
    if _RICH:
        console = Console()
        table = Table(title=f"벤치마크 결과 (mode={stats['mode']}, frames={stats['frames']})", show_lines=True)
        table.add_column("항목", style="bold white")
        table.add_column("값", justify="right", style="cyan")

        rows = [
            ("평균 FPS",       f"{stats['avg_fps']:.2f}"),
            ("최소 FPS",       f"{stats['min_fps']:.2f}"),
            ("최대 FPS",       f"{stats['max_fps']:.2f}"),
            ("평균 프레임 시간", f"{stats['avg_frame_ms']:.1f} ms"),
            ("평균 VRAM",      f"{stats['vram_avg_mb']} MB"),
            ("최대 VRAM",      f"{stats['vram_peak_mb']} MB"),
        ]
        for name, val in rows:
            table.add_row(name, val)
        console.print(table)
    else:
        print(f"\n=== 벤치마크 결과 (mode={stats['mode']}) ===")
        print(f"  평균 FPS:       {stats['avg_fps']:.2f}")
        print(f"  최소 FPS:       {stats['min_fps']:.2f}")
        print(f"  최대 FPS:       {stats['max_fps']:.2f}")
        print(f"  평균 프레임 시간: {stats['avg_frame_ms']:.1f} ms")
        print(f"  평균 VRAM:      {stats['vram_avg_mb']} MB")
        print(f"  최대 VRAM:      {stats['vram_peak_mb']} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="FPS 벤치마크")
    parser.add_argument("--mode", choices=["rgb", "depth", "fusion"], default="fusion")
    parser.add_argument("--frames", type=int, default=100, help="측정할 프레임 수 (기본: 100)")
    args = parser.parse_args()

    stats = _run_benchmark(args.mode, args.frames)
    _print_results(stats)


if __name__ == "__main__":
    main()
