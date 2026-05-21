"""모델 가중치 자동 다운로드 스크립트.

Usage:
    python scripts/download_weights.py --model all
    python scripts/download_weights.py --model yolov11
    python scripts/download_weights.py --model rtdetrv2
    python scripts/download_weights.py --model grounding_dino
    python scripts/download_weights.py --model pointpillars
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
    _RICH = True
except ImportError:
    _RICH = False

console = Console() if _RICH else None


_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")

_GDINO_URLS = {
    "groundingdino_swint_ogc.pth": (
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        "v0.1.0-alpha/groundingdino_swint_ogc.pth"
    ),
    "GroundingDINO_SwinT_OGC.py": (
        "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/"
        "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    ),
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _print(msg: str, style: str = "") -> None:
    if _RICH:
        console.print(msg, style=style)
    else:
        print(msg)


def _download_file(url: str, dest: str) -> None:
    import requests

    _ensure_dir(os.path.dirname(dest))

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    if _RICH:
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"[cyan]{os.path.basename(dest)}", total=total or None)
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress.advance(task, len(chunk))
    else:
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    _print(f"[green]완료:[/green] {dest}" if _RICH else f"완료: {dest}")


def download_yolov11() -> None:
    _print("[bold cyan]YOLOv11 가중치 다운로드 중...[/bold cyan]" if _RICH else "YOLOv11 가중치 다운로드 중...")
    try:
        from ultralytics import YOLO
        dest_dir = os.path.join(_WEIGHTS_DIR, "yolo")
        _ensure_dir(dest_dir)
        model = YOLO("yolo11x.pt")
        src = "yolo11x.pt"
        dest = os.path.join(dest_dir, "yolo11x.pt")
        if os.path.exists(src) and not os.path.exists(dest):
            import shutil
            shutil.move(src, dest)
        _print("[green]YOLOv11 다운로드 완료.[/green]" if _RICH else "YOLOv11 다운로드 완료.")
    except ImportError:
        _print("[red]ultralytics 미설치: pip install ultralytics[/red]" if _RICH else "ultralytics 미설치: pip install ultralytics")


def download_grounding_dino() -> None:
    _print("[bold cyan]Grounding DINO 가중치 다운로드 중...[/bold cyan]" if _RICH else "Grounding DINO 가중치 다운로드 중...")
    dest_dir = os.path.join(_WEIGHTS_DIR, "grounding_dino")
    for filename, url in _GDINO_URLS.items():
        dest = os.path.join(dest_dir, filename)
        if os.path.exists(dest):
            _print(f"[yellow]스킵 (이미 존재): {dest}[/yellow]" if _RICH else f"스킵: {dest}")
            continue
        try:
            _download_file(url, dest)
        except Exception as e:
            _print(f"[red]다운로드 실패 {filename}: {e}[/red]" if _RICH else f"다운로드 실패 {filename}: {e}")


def download_pointpillars() -> None:
    _print(
        "[yellow]PointPillars 자동 다운로드는 지원되지 않습니다.[/yellow]\n"
        "KITTI pretrained 가중치를 수동으로 다운로드하세요:\n"
        "  https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md\n"
        "다운로드 후 weights/pointpillars/ 디렉토리에 배치하세요."
        if _RICH else
        "PointPillars 자동 다운로드는 지원되지 않습니다.\n"
        "https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md\n"
        "weights/pointpillars/ 디렉토리에 수동 배치 필요."
    )


def download_rtdetrv2() -> None:
    _print("[bold cyan]RT-DETRv2 가중치 다운로드 중 (HuggingFace Hub)...[/bold cyan]" if _RICH else "RT-DETRv2 가중치 다운로드 중 (HuggingFace Hub)...")
    hf_model_id = "PekingU/rtdetr_v2_r50vd"
    dest_dir = os.path.join(_WEIGHTS_DIR, "rtdetrv2")
    cache_marker = os.path.join(dest_dir, ".hf_cached")
    if os.path.exists(cache_marker):
        _print(f"[yellow]스킵 (이미 존재): {dest_dir}[/yellow]" if _RICH else f"스킵: {dest_dir}")
        return
    try:
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        _ensure_dir(dest_dir)
        _print(f"  모델 ID: {hf_model_id}" if not _RICH else f"  [dim]모델 ID: {hf_model_id}[/dim]")
        RTDetrImageProcessor.from_pretrained(hf_model_id)
        RTDetrV2ForObjectDetection.from_pretrained(hf_model_id)
        open(cache_marker, "w").close()
        _print("[green]RT-DETRv2 다운로드 완료 (~/.cache/huggingface/).[/green]" if _RICH else "RT-DETRv2 다운로드 완료 (~/.cache/huggingface/).")
    except Exception as e:
        _print(f"[red]RT-DETRv2 다운로드 실패: {e}[/red]" if _RICH else f"RT-DETRv2 다운로드 실패: {e}")


_DOWNLOADERS = {
    "yolov11": download_yolov11,
    "rtdetrv2": download_rtdetrv2,
    "grounding_dino": download_grounding_dino,
    "pointpillars": download_pointpillars,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 가중치 자동 다운로드")
    parser.add_argument(
        "--model",
        choices=["all", "yolov11", "rtdetrv2", "grounding_dino", "pointpillars"],
        default="all",
        help="다운로드할 모델 (기본: all)",
    )
    args = parser.parse_args()

    _ensure_dir(_WEIGHTS_DIR)

    targets = list(_DOWNLOADERS.keys()) if args.model == "all" else [args.model]
    for name in targets:
        _DOWNLOADERS[name]()

    _print("[bold green]모든 다운로드 작업 완료.[/bold green]" if _RICH else "모든 다운로드 작업 완료.")


if __name__ == "__main__":
    main()
