from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass, field
from types import TracebackType

from rtsp_supervisor import ExponentialBackoff, FrameSource, StreamSupervisor, SupervisorConfig
from rtsp_supervisor.opencv import OpenCVSource
from typed_mqtt_bus import Bus, InMemoryBroker, Transport
from typed_mqtt_bus.mqtt import AiomqttTransport

from .config import AppConfig, CameraConfig, MqttConfig, RestartConfig
from .detectors import Frame
from .pipeline import CameraPipeline
from .rules import LineRule, Rule, ZoneRule
from .sinks import BusSink, DeliveryStats, EventSink, FanOut, RepositorySink
from .status import CameraStatus, RuleStatus
from .storage import SqlEventRepository

HEALTHY_STALENESS = 5.0

logger = logging.getLogger(__name__)

SourceFactory = Callable[[CameraConfig], FrameSource[Frame]]
TransportFactory = Callable[[MqttConfig], AbstractAsyncContextManager[Transport]]


def open_camera(camera: CameraConfig) -> FrameSource[Frame]:
    return OpenCVSource(camera.url.get_secret_value())


def connect_mqtt(config: MqttConfig) -> AbstractAsyncContextManager[Transport]:
    return AiomqttTransport(
        config.host,
        config.port,
        username=config.username,
        password=config.password.get_secret_value() if config.password else None,
        connect_timeout=30.0,
    )


@dataclass
class Camera:
    config: CameraConfig
    supervisor: StreamSupervisor[Frame]
    pipeline: CameraPipeline
    restart: RestartConfig
    restarts: int = field(default=0, init=False)
    running: bool = field(default=True, init=False)

    async def run(self) -> None:
        backoff = ExponentialBackoff(self.restart.initial_delay, self.restart.max_delay)
        while True:
            started = time.monotonic()
            self.running = True
            try:
                await self.pipeline.run(self.supervisor)
                return
            except Exception:
                logger.exception("pipeline for camera %s crashed", self.config.id)
            finally:
                self.running = False
            if time.monotonic() - started >= self.restart.stable_after:
                backoff.reset()
            delay = backoff.next_delay()
            logger.info("restarting pipeline for camera %s in %.3fs", self.config.id, delay)
            await asyncio.sleep(delay)
            self.restarts += 1

    def status(self) -> CameraStatus:
        stream, stats = self.supervisor.stats, self.pipeline.stats
        return CameraStatus(
            id=self.config.id,
            stream_state=stream.state.value,
            healthy=self.supervisor.is_healthy(HEALTHY_STALENESS) and self.running,
            pipeline_running=self.running,
            pipeline_restarts=self.restarts,
            frames_read=stream.frames_read,
            frames_dropped=stream.frames_dropped,
            reconnect_attempts=stream.reconnect_attempts,
            last_error=stream.last_error,
            frames_processed=stats.frames_processed,
            frames_skipped=stats.frames_skipped,
            detector_errors=stats.detector_errors,
            events_emitted=stats.events_emitted,
            active_tracks=stats.active_tracks,
            inference_ms=round(stats.inference_ms, 2) if stats.inference_ms is not None else None,
            rules=[_rule_status(rule) for rule in self.pipeline.rules],
        )


class Runtime:
    def __init__(
        self,
        config: AppConfig,
        *,
        source_factory: SourceFactory = open_camera,
        transport_factory: TransportFactory = connect_mqtt,
    ) -> None:
        self._config = config
        self._source_factory = source_factory
        self._transport_factory = transport_factory
        self._stack = AsyncExitStack()
        self._cameras: list[Camera] = []
        self._fan_out: FanOut | None = None
        self.repository: SqlEventRepository
        self.bus: Bus

    async def __aenter__(self) -> Runtime:
        try:
            await self._start()
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._stack.aclose()

    def camera_statuses(self) -> list[CameraStatus]:
        return [camera.status() for camera in self._cameras]

    def delivery_stats(self) -> dict[str, DeliveryStats]:
        return self._fan_out.stats if self._fan_out is not None else {}

    async def _start(self) -> None:
        self.repository = SqlEventRepository.from_url(self._config.database_url.get_secret_value())
        self._stack.push_async_callback(self.repository.close)
        await self.repository.create_schema()
        transport = await self._stack.enter_async_context(self._open_transport())
        self.bus = await self._stack.enter_async_context(Bus(transport))
        sinks: dict[str, EventSink] = {
            "database": RepositorySink(self.repository),
            "mqtt": BusSink(self.bus),
        }
        fan_out = await self._stack.enter_async_context(FanOut(sinks))
        self._fan_out = fan_out
        for camera in self._config.cameras:
            self._cameras.append(await self._start_camera(camera, fan_out))

    def _open_transport(self) -> AbstractAsyncContextManager[Transport]:
        if self._config.mqtt is None:
            return contextlib.nullcontext(InMemoryBroker().client())
        return self._transport_factory(self._config.mqtt)

    async def _start_camera(self, camera: CameraConfig, fan_out: FanOut) -> Camera:
        supervisor: StreamSupervisor[Frame] = StreamSupervisor(
            lambda: self._source_factory(camera), config=SupervisorConfig(read_timeout=10.0)
        )
        await self._stack.enter_async_context(supervisor)
        pipeline = CameraPipeline(
            camera.id,
            self._config.detector_for(camera).build(),
            self._config.tracker.build(),
            [rule.build() for rule in camera.rules],
            fan_out,
            max_fps=camera.max_fps,
        )
        runner = Camera(camera, supervisor, pipeline, self._config.pipeline_restart)
        task = asyncio.create_task(runner.run(), name=f"pipeline-{camera.id}")
        self._stack.push_async_callback(_cancel, task)
        return runner


async def _cancel(task: asyncio.Task[None]) -> None:
    task.cancel()
    await asyncio.wait({task})


def _rule_status(rule: Rule) -> RuleStatus:
    if isinstance(rule, ZoneRule):
        return RuleStatus(id=rule.id, type="zone", occupancy=rule.occupancy)
    if isinstance(rule, LineRule):
        return RuleStatus(id=rule.id, type="line", counts=rule.counts)
    return RuleStatus(id=rule.id, type=type(rule).__name__)
