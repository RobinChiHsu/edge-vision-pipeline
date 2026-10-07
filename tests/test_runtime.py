from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from pathlib import Path

import pytest
from rtsp_supervisor import FrameSource
from typed_mqtt_bus import InMemoryBroker, Transport
from typed_mqtt_bus.mqtt import AiomqttTransport

from edge_vision.config import AppConfig, CameraConfig, MqttConfig
from edge_vision.detectors import Frame
from edge_vision.rules import RuleHit
from edge_vision.runtime import Runtime, _rule_status, connect_mqtt, open_camera
from edge_vision.storage import EventQuery
from edge_vision.tracking import Track
from tests.helpers import MovingSquareSource, wait_until


def make_config(tmp_path: Path, **overrides: object) -> AppConfig:
    return AppConfig.model_validate(
        {
            "database_url": f"sqlite+aiosqlite:///{tmp_path / 'events.db'}",
            "detector": {"type": "motion", "min_area": 200, "warmup_frames": 10},
            "tracker": {"min_hits": 1},
            "cameras": [
                {
                    "id": "lab",
                    "url": "rtsp://viewer:secret@camera.local/stream",
                    "max_fps": None,
                    "rules": [
                        {"type": "line", "id": "door", "start": [100, 0], "end": [100, 300]},
                        {
                            "type": "zone",
                            "id": "bench",
                            "polygon": [[0, 0], [60, 0], [60, 120], [0, 120]],
                        },
                    ],
                }
            ],
            **overrides,
        }
    )


def synthetic_camera(camera: CameraConfig) -> FrameSource[Frame]:
    return MovingSquareSource()


async def test_detects_events_and_reports_status(tmp_path: Path) -> None:
    async with Runtime(make_config(tmp_path), source_factory=synthetic_camera) as runtime:
        await wait_until(lambda: runtime.camera_statuses()[0].events_emitted >= 2, within=6.0)
        await wait_until(lambda: runtime.delivery_stats()["database"].delivered >= 2)

        [status] = runtime.camera_statuses()
        page = await runtime.repository.list(EventQuery(camera_id="lab"))

    assert {event.kind for event in page.items} == {"line_crossed", "zone_entered"}
    assert status.id == "lab"
    assert status.pipeline_running
    assert status.frames_processed > 0
    assert {rule.id: rule.type for rule in status.rules} == {"door": "line", "bench": "zone"}
    assert status.rules[0].counts == {"left": 1, "right": 0}


async def test_uses_mqtt_transport_when_configured(tmp_path: Path) -> None:
    broker = InMemoryBroker()
    opened: list[MqttConfig] = []

    def fake_mqtt(config: MqttConfig) -> contextlib.nullcontext[Transport]:
        opened.append(config)
        return contextlib.nullcontext(broker.client())

    config = make_config(tmp_path, mqtt={"host": "broker.local"})
    async with Runtime(config, source_factory=synthetic_camera, transport_factory=fake_mqtt) as rt:
        await wait_until(lambda: rt.delivery_stats()["mqtt"].delivered >= 1, within=6.0)

    assert [item.host for item in opened] == ["broker.local"]
    assert broker.history[0].topic == "edge-vision/cameras/lab/events"


async def test_failed_startup_releases_resources(tmp_path: Path) -> None:
    def unreachable(config: MqttConfig) -> contextlib.nullcontext[Transport]:
        raise ConnectionError("broker down")

    config = make_config(tmp_path, mqtt={"host": "broker.local"})

    with pytest.raises(ConnectionError):
        async with Runtime(config, transport_factory=unreachable):
            pass


async def test_crashed_pipeline_is_logged_and_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def explode(self: object, feed: object) -> None:
        raise RuntimeError("pipeline bug")

    monkeypatch.setattr("edge_vision.pipeline.CameraPipeline.run", explode)

    with caplog.at_level(logging.ERROR):
        async with Runtime(make_config(tmp_path), source_factory=synthetic_camera) as runtime:
            await wait_until(lambda: not runtime.camera_statuses()[0].pipeline_running)
            [status] = runtime.camera_statuses()

    assert not status.healthy
    assert "pipeline for camera lab crashed" in caplog.text


def test_default_factories_build_real_clients(tmp_path: Path) -> None:
    camera = make_config(tmp_path).cameras[0]

    assert type(open_camera(camera)).__name__ == "OpenCVSource"
    mqtt = connect_mqtt(MqttConfig(host="broker.local", username="svc", password="pw"))
    assert isinstance(mqtt, AiomqttTransport)
    assert isinstance(connect_mqtt(MqttConfig(host="broker.local")), AiomqttTransport)


def test_unknown_rule_type_still_has_a_status() -> None:
    class Opaque:
        id = "custom"

        def evaluate(self, tracks: Sequence[Track], now: float) -> list[RuleHit]:
            return []

    assert _rule_status(Opaque()).type == "Opaque"


def test_delivery_stats_empty_before_start(tmp_path: Path) -> None:
    assert Runtime(make_config(tmp_path)).delivery_stats() == {}
