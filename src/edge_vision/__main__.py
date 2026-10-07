from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from .api import create_app
from .config import load_config
from .runtime import Runtime


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="edge-vision", description="Run the video analytics service and its HTTP API."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("EDGE_VISION_CONFIG", "config/local.yaml")),
        help="path to the YAML configuration (default: $EDGE_VISION_CONFIG or config/local.yaml)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    return parser.parse_args(argv)


def build_app(config_path: Path) -> FastAPI:
    config = load_config(config_path)
    return create_app(lambda: Runtime(config))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    uvicorn.run(build_app(args.config), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
