from pathlib import Path

import pytest
from pydantic import ValidationError

from edge_vision.config import (
    AppConfig,
    CameraConfig,
    MissingEnvironmentVariable,
    MotionDetectorConfig,
    YoloDetectorConfig,
    expand_environment,
    load_config,
)
from edge_vision.detectors.motion import MotionDetector
from edge_vision.rules import LineRule, ZoneRule
from edge_vision.tracking import IouTracker

EXAMPLE = """
database_url: postgresql+asyncpg://edge:${DB_PASSWORD}@db/edge
mqtt:
  host: ${MQTT_HOST:-localhost}
detector:
  type: motion
  min_area: 600
cameras:
  - id: gate
    url: rtsp://viewer:${CAMERA_PASSWORD}@10.0.0.5/stream1
    max_fps: 8
    rules:
      - type: line
        id: entrance
        start: [320, 0]
        end: [320, 480]
        direction: left
      - type: zone
        id: restricted
        polygon: [[0, 0], [100, 0], [100, 100]]
        labels: [person]
        min_dwell: 2.5
  - id: dock
    url: rtsp://10.0.0.6/stream1
    detector:
      type: yolo
      model: models/yolo.onnx
      labels: [person, car]
      classes: [car]
"""

ENVIRONMENT = {"DB_PASSWORD": "db-secret", "CAMERA_PASSWORD": "cam-secret"}


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def minimal(**overrides: object) -> dict[str, object]:
    return {
        "database_url": "sqlite+aiosqlite://",
        "cameras": [{"id": "gate", "url": "rtsp://cam"}],
        **overrides,
    }


def test_loads_full_example_with_environment(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, EXAMPLE), ENVIRONMENT)

    gate, dock = config.cameras
    assert config.database_url.get_secret_value() == "postgresql+asyncpg://edge:db-secret@db/edge"
    assert config.mqtt is not None
    assert config.mqtt.host == "localhost"
    assert gate.url.get_secret_value() == "rtsp://viewer:cam-secret@10.0.0.5/stream1"
    assert gate.max_fps == 8
    assert config.detector_for(gate) == MotionDetectorConfig(min_area=600)
    assert isinstance(config.detector_for(dock), YoloDetectorConfig)
    assert dock.max_fps == 5.0


def test_secrets_stay_out_of_repr(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, EXAMPLE), ENVIRONMENT)

    assert "secret" not in repr(config)
    assert "secret" not in str(config.model_dump())


def test_builds_runtime_objects(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, EXAMPLE), ENVIRONMENT)
    gate = config.cameras[0]

    line, zone = (rule.build() for rule in gate.rules)

    assert isinstance(line, LineRule)
    assert line.id == "entrance"
    assert isinstance(zone, ZoneRule)
    assert zone.id == "restricted"
    assert isinstance(config.detector_for(gate).build(), MotionDetector)
    assert isinstance(config.tracker.build(), IouTracker)
    assert line.counts == {"left": 0, "right": 0}
    assert gate.rules[0].type == "line"


def test_yolo_config_builds_onnx_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[str, list[str]]] = []

    class FakeDetector:
        def __init__(self, model: str, labels: list[str], **_: object) -> None:
            built.append((model, labels))

    monkeypatch.setattr("edge_vision.detectors.yolo.OnnxYoloDetector", FakeDetector)

    YoloDetectorConfig(type="yolo", model=Path("m.onnx"), labels=["person"]).build()

    assert built == [("m.onnx", ["person"])]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cameras": []}, "at least 1"),
        ({"cameras": [{"id": "gate", "url": "a"}, {"id": "gate", "url": "b"}]}, "duplicate camera"),
        ({"cameras": [{"id": "Gate/1", "url": "a"}]}, "pattern"),
        ({"cameras": [{"id": "gate", "url": "a", "max_fps": 0}]}, "greater than 0"),
        ({"unknown": True}, "Extra inputs"),
    ],
)
def test_rejects_invalid_app_config(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        AppConfig.model_validate(minimal(**overrides))


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ([{"type": "zone", "id": "z", "polygon": [[0, 0], [1, 1]]}], "at least 3"),
        ([{"type": "line", "id": "l", "start": [1, 1], "end": [1, 1]}], "zero length"),
        ([{"type": "circle", "id": "c"}], "does not match any of the expected tags"),
        (
            [
                {"type": "line", "id": "same", "start": [0, 0], "end": [1, 1]},
                {"type": "zone", "id": "same", "polygon": [[0, 0], [1, 0], [1, 1]]},
            ],
            "duplicate rule id",
        ),
    ],
)
def test_rejects_invalid_rules(rules: list[dict[str, object]], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        CameraConfig.model_validate({"id": "gate", "url": "rtsp://cam", "rules": rules})


class TestExpandEnvironment:
    def test_substitutes_variables_and_defaults(self) -> None:
        text = "${A} ${B:-fallback} ${C:-}"

        assert expand_environment(text, {"A": "1"}) == "1 fallback "

    def test_set_variable_wins_over_default(self) -> None:
        assert expand_environment("${A:-x}", {"A": "real"}) == "real"

    def test_missing_variable_without_default_is_an_error(self) -> None:
        with pytest.raises(MissingEnvironmentVariable, match="DB_PASSWORD"):
            expand_environment("${DB_PASSWORD}", {})

    def test_reads_process_environment_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EDGE_VISION_EXAMPLE", "from-env")

        assert expand_environment("${EDGE_VISION_EXAMPLE}") == "from-env"
