"""Azure Kinect 캘리브레이션 확인 및 저장.

Usage:
    python scripts/calibration_check.py
    python scripts/calibration_check.py --save calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


def _extract_intrinsics(cam_cal) -> dict:
    """pyk4a 카메라 캘리브레이션 객체에서 내부 파라미터를 추출합니다."""
    params = cam_cal.intrinsics.parameters.param
    return {
        "fx": float(params.fx),
        "fy": float(params.fy),
        "cx": float(params.cx),
        "cy": float(params.cy),
        "k1": float(params.k1),
        "k2": float(params.k2),
        "k3": float(params.k3),
        "k4": float(params.k4),
        "k5": float(params.k5),
        "k6": float(params.k6),
        "p1": float(params.p1),
        "p2": float(params.p2),
        "width": int(cam_cal.resolution_width),
        "height": int(cam_cal.resolution_height),
    }


def _print_intrinsics(title: str, data: dict) -> None:
    if _RICH:
        console = Console()
        table = Table(title=title, show_lines=True)
        table.add_column("파라미터", style="bold white")
        table.add_column("값", justify="right", style="cyan")
        for k, v in data.items():
            table.add_row(str(k), f"{v}")
        console.print(table)
    else:
        print(f"\n=== {title} ===")
        for k, v in data.items():
            print(f"  {k:12s}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Azure Kinect 캘리브레이션 확인")
    parser.add_argument("--save", metavar="PATH", default=None, help="저장할 JSON 파일 경로")
    args = parser.parse_args()

    try:
        import pyk4a
        from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS
    except ImportError:
        print("pyk4a 미설치: pip install pyk4a==1.4.1")
        sys.exit(1)

    config = Config(
        color_resolution=ColorResolution.RES_1080P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        camera_fps=FPS.FPS_30,
    )

    try:
        device = PyK4A(config=config)
        device.start()
    except Exception as e:
        print(f"장치 연결 실패: {e}")
        print("  확인: USB 3.0 포트, Azure Kinect SDK v1.4.1 설치 여부")
        sys.exit(1)

    try:
        cal = device.calibration
        color_data = _extract_intrinsics(cal.get_camera_calibration(pyk4a.calibration.CalibrationType.COLOR))
        depth_data = _extract_intrinsics(cal.get_camera_calibration(pyk4a.calibration.CalibrationType.DEPTH))
    except Exception as e:
        print(f"캘리브레이션 취득 실패: {e}")
        device.stop()
        sys.exit(1)
    finally:
        device.stop()

    _print_intrinsics("Color Camera 내부 파라미터", color_data)
    _print_intrinsics("Depth Camera 내부 파라미터", depth_data)

    result = {"color": color_data, "depth": depth_data}

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        msg = f"캘리브레이션 저장 완료: {args.save}"
        if _RICH:
            Console().print(f"[green]{msg}[/green]")
        else:
            print(msg)


if __name__ == "__main__":
    main()
