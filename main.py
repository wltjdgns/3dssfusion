import argparse
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from pipeline.rgb_pipeline import RGBPipeline
from pipeline.depth_pipeline import DepthPipeline
from pipeline.fusion_pipeline import FusionPipeline

PIPELINE_MAP = {
    "rgb": RGBPipeline,
    "depth": DepthPipeline,
    "fusion": FusionPipeline,
}


class ConfigLoader:
    def __init__(self, config_path: str):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(path) as f:
            self._cfg = yaml.safe_load(f)

    def override_mode(self, mode: str | None) -> None:
        if mode is not None:
            self._cfg["mode"] = mode

    def override_device(self, device: str | None) -> None:
        if device is not None:
            self._cfg["gpu"]["rgb_device"] = device
            self._cfg["gpu"]["depth_device"] = device

    def __getitem__(self, key: str):
        return self._cfg[key]

    def get(self, key: str, default=None):
        return self._cfg.get(key, default)

    @property
    def raw(self) -> dict:
        return self._cfg


def _build_banner() -> Panel:
    text = Text()
    text.append("Azure Kinect DK  Real-Time Detection System\n", style="bold cyan")
    text.append("v0.1.0   Modes: rgb  |  depth  |  fusion\n", style="dim white")
    text.append("Press Ctrl+C to stop", style="italic dim")
    return Panel(text, border_style="cyan", expand=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Azure Kinect real-time object detection"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["rgb", "depth", "fusion"],
        default=None,
        help="Override config mode",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override CUDA device for all models, e.g. cuda:0",
    )
    return parser.parse_args()


def main() -> None:
    console = Console()
    console.print(_build_banner())

    args = _parse_args()

    try:
        cfg = ConfigLoader(args.config)
    except FileNotFoundError as e:
        console.print(f"[red][ERROR] {e}[/red]")
        sys.exit(1)

    cfg.override_mode(args.mode)
    cfg.override_device(args.device)

    mode: str = cfg["mode"]
    if mode not in PIPELINE_MAP:
        console.print(f"[red][ERROR] Unknown mode '{mode}'. Choose: {list(PIPELINE_MAP)}[/red]")
        sys.exit(1)

    console.print(f"[green]Mode:[/green] [bold]{mode}[/bold]  |  "
                  f"[green]RGB device:[/green] {cfg['gpu']['rgb_device']}  |  "
                  f"[green]Depth device:[/green] {cfg['gpu']['depth_device']}")

    pipeline = PIPELINE_MAP[mode](cfg)

    try:
        pipeline.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutdown requested — stopping pipeline...[/yellow]")
    finally:
        pipeline.cleanup()
        console.print("[green]Pipeline stopped cleanly.[/green]")


if __name__ == "__main__":
    main()
