from __future__ import annotations

import asyncio
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
from edge_vision.pipeline import CameraPipeline, FrameFeed
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


def restart_delays(caplog: pytest.LogCaptureFixture) -> list[float]:
    prefix = "restarting pipeline for camera lab in "
    return [
        float(record.getMessage().removeprefix(prefix).removesuffix("s"))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


async def test_crashed_pipeline_restarts_and_recovers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_run = CameraPipeline.run
    crashes = 0

    async def crash_twice(self: CameraPipeline, feed: FrameFeed) -> None:
        nonlocal crashes
        if crashes < 2:
            crashes += 1
            raise RuntimeError("pipeline bug")
        await real_run(self, feed)

    monkeypatch.setattr(CameraPipeline, "run", crash_twice)
    config = make_config(tmp_path, pipeline_restart={"initial_delay": 0.01, "max_delay": 1})

    with caplog.at_level(logging.INFO, logger="edge_vision.runtime"):
        async with Runtime(config, source_factory=synthetic_camera) as runtime:
            await wait_until(lambda: runtime.camera_statuses()[0].events_emitted >= 1, within=6.0)
            [status] = runtime.camera_statuses()

    assert status.pipeline_restarts == 2
    assert status.pipeline_running
    assert caplog.text.count("pipeline for camera lab crashed") == 2


async def test_restart_delay_grows_up_to_the_maximum(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def explode(self: CameraPipeline, feed: FrameFeed) -> None:
        raise RuntimeError("pipeline bug")

    monkeypatch.setattr(CameraPipeline, "run", explode)
    config = make_config(tmp_path, pipeline_restart={"initial_delay": 0.01, "max_delay": 0.04})

    with caplog.at_level(logging.INFO, logger="edge_vision.runtime"):
        async with Runtime(config, source_factory=synthetic_camera) as runtime:
            await wait_until(lambda: runtime.camera_statuses()[0].pipeline_restarts >= 4)
            [status] = runtime.camera_statuses()

    delays = restart_delays(caplog)[:4]
    assert [round(delay, 2) for delay in delays] == [0.01, 0.02, 0.04, 0.04]
    assert not status.healthy


async def test_stable_run_resets_restart_delay(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def explode(self: CameraPipeline, feed: FrameFeed) -> None:
        raise RuntimeError("pipeline bug")

    monkeypatch.setattr(CameraPipeline, "run", explode)
    restart = {"initial_delay": 0.01, "max_delay": 1, "stable_after": 0}
    config = make_config(tmp_path, pipeline_restart=restart)

    with caplog.at_level(logging.INFO, logger="edge_vision.runtime"):
        async with Runtime(config, source_factory=synthetic_camera) as runtime:
            await wait_until(lambda: runtime.camera_statuses()[0].pipeline_restarts >= 3)

    assert [round(delay, 2) for delay in restart_delays(caplog)[:3]] == [0.01, 0.01, 0.01]


async def test_pipeline_is_not_restarted_after_a_clean_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def finish(self: CameraPipeline, feed: FrameFeed) -> None:
        return None

    monkeypatch.setattr(CameraPipeline, "run", finish)
    config = make_config(tmp_path, pipeline_restart={"initial_delay": 0.01})

    async with Runtime(config, source_factory=synthetic_camera) as runtime:
        await asyncio.sleep(0.1)
        [status] = runtime.camera_statuses()

    assert not status.pipeline_running
    assert status.pipeline_restarts == 0


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
