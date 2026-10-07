from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI

from edge_vision.__main__ import build_app, main, parse_args

CONFIG = """
database_url: sqlite+aiosqlite:///{db}
cameras:
  - id: gate
    url: rtsp://camera.local/stream
"""


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG.format(db=tmp_path / "events.db"))
    return path


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDGE_VISION_CONFIG", raising=False)

    args = parse_args([])

    assert (args.config, args.host, args.port) == (Path("config/local.yaml"), "127.0.0.1", 8000)


def test_config_path_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGE_VISION_CONFIG", "/etc/edge-vision.yaml")

    assert parse_args([]).config == Path("/etc/edge-vision.yaml")


def test_build_app_validates_config_eagerly(tmp_path: Path) -> None:
    assert isinstance(build_app(write_config(tmp_path)), FastAPI)

    broken = tmp_path / "broken.yaml"
    broken.write_text("cameras: []\n")
    with pytest.raises(ValueError, match="database_url"):
        build_app(broken)


def test_main_starts_uvicorn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append({"app": app, **kwargs}))

    main(["--config", str(write_config(tmp_path)), "--host", "0.0.0.0", "--port", "9000"])

    [call] = calls
    assert isinstance(call["app"], FastAPI)
    assert (call["host"], call["port"], call["log_level"]) == ("0.0.0.0", 9000, "info")
