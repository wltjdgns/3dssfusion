"""설치 검증 스크립트.

Usage:
    python scripts/verify_install.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False
    print("[경고] rich 미설치. pip install rich")


@dataclass
class CheckResult:
    name: str
    status: str   # "PASS" | "FAIL" | "SKIP"
    detail: str
    fix_hint: str = ""


def _check_python_version() -> CheckResult:
    v = sys.version_info
    if v >= (3, 10):
        return CheckResult("Python >= 3.10", "PASS", f"{v.major}.{v.minor}.{v.micro}")
    return CheckResult(
        "Python >= 3.10", "FAIL",
        f"현재: {v.major}.{v.minor}.{v.micro}",
        "Python 3.10 이상 설치 필요. https://www.python.org/downloads/",
    )


def _check_pytorch_cuda() -> CheckResult:
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        ver = torch.__version__
        if cuda_ok:
            dev = torch.cuda.get_device_name(0)
            return CheckResult("PyTorch + CUDA", "PASS", f"PyTorch {ver} | {dev}")
        return CheckResult(
            "PyTorch + CUDA", "FAIL",
            f"PyTorch {ver} (CUDA 불가)",
            "CUDA 지원 PyTorch 설치: https://pytorch.org/get-started/locally/",
        )
    except ImportError:
        return CheckResult(
            "PyTorch + CUDA", "FAIL", "미설치",
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
        )


def _check_pyk4a() -> CheckResult:
    try:
        import pyk4a
        ver = getattr(pyk4a, "__version__", "unknown")
        return CheckResult("pyk4a", "PASS", f"버전: {ver}")
    except ImportError:
        return CheckResult(
            "pyk4a", "FAIL", "미설치",
            "pip install pyk4a==1.4.1 후 Azure Kinect SDK v1.4.1 설치 필요",
        )


def _check_kinect_device() -> CheckResult:
    try:
        import pyk4a
        devices = pyk4a.connected_device_count()
        if devices > 0:
            return CheckResult("Azure Kinect 장치", "PASS", f"{devices}개 연결됨")
        return CheckResult("Azure Kinect 장치", "SKIP", "연결된 장치 없음 (실행 시 필요)")
    except Exception:
        return CheckResult("Azure Kinect 장치", "SKIP", "pyk4a 로드 불가로 확인 생략")


def _check_import(pkg_name: str, display: str, fix: str) -> CheckResult:
    try:
        mod = __import__(pkg_name)
        ver = getattr(mod, "__version__", "ok")
        return CheckResult(display, "PASS", f"버전: {ver}")
    except ImportError:
        return CheckResult(display, "FAIL", "미설치", fix)


def _check_ultralytics() -> CheckResult:
    return _check_import("ultralytics", "ultralytics", "pip install ultralytics")


def _check_transformers() -> CheckResult:
    return _check_import(
        "transformers", "transformers",
        "pip install transformers",
    )


def _check_groundingdino() -> CheckResult:
    return _check_import(
        "groundingdino", "groundingdino",
        "pip install groundingdino-py  또는 소스 빌드: "
        "https://github.com/IDEA-Research/GroundingDINO",
    )


def _check_spconv() -> CheckResult:
    return _check_import(
        "spconv", "spconv (OpenPCDet)",
        "pip install spconv-cu118  (CUDA 버전에 맞게 선택)",
    )


def _check_open3d() -> CheckResult:
    return _check_import("open3d", "open3d", "pip install open3d")


def _check_ensemble_boxes() -> CheckResult:
    return _check_import(
        "ensemble_boxes", "ensemble_boxes",
        "pip install ensemble-boxes",
    )


_CHECKS: list[Callable[[], CheckResult]] = [
    _check_python_version,
    _check_pytorch_cuda,
    _check_pyk4a,
    _check_kinect_device,
    _check_ultralytics,
    _check_transformers,
    _check_groundingdino,
    _check_spconv,
    _check_open3d,
    _check_ensemble_boxes,
]


def main() -> None:
    results = [fn() for fn in _CHECKS]

    if _RICH:
        console = Console()
        table = Table(title="Azure Kinect 설치 검증 결과", show_lines=True)
        table.add_column("항목", style="bold white", min_width=25)
        table.add_column("상태", justify="center", min_width=6)
        table.add_column("상세", min_width=35)
        table.add_column("해결 방법", style="dim", min_width=40)

        for r in results:
            color = {"PASS": "green", "FAIL": "red", "SKIP": "yellow"}.get(r.status, "white")
            table.add_row(r.name, f"[{color}]{r.status}[/{color}]", r.detail, r.fix_hint)

        console.print(table)

        failed = [r for r in results if r.status == "FAIL"]
        if failed:
            console.print(f"\n[red]{len(failed)}개 항목 실패.[/red] 위 해결 방법을 참고하세요.")
        else:
            console.print("\n[green bold]모든 검증 통과![/green bold]")
    else:
        print("\n=== Azure Kinect 설치 검증 결과 ===")
        for r in results:
            print(f"[{r.status:4s}] {r.name:30s} {r.detail}")
            if r.fix_hint:
                print(f"       해결: {r.fix_hint}")
        failed = [r for r in results if r.status == "FAIL"]
        if failed:
            print(f"\n{len(failed)}개 항목 실패.")
        else:
            print("\n모든 검증 통과!")


if __name__ == "__main__":
    main()
