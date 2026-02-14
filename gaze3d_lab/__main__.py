from __future__ import annotations

import argparse

from gaze3d_lab.app.runner import run_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gaze3d_lab",
        description="Real-time 3D gaze ray + scene intersection visualizer",
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "webcam", "recorded"],
        default="synthetic",
        help="Input stream mode",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_app(mode=args.mode, config_path=args.config)


if __name__ == "__main__":
    raise SystemExit(main())
