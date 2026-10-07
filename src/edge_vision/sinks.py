from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol

from typed_mqtt_bus import Bus, Topic

from .events import Event
from .storage import EventRepository

EVENTS_TOPIC = Topic("edge-vision/cameras/{camera_id}/events", Event)

logger = logging.getLogger(__name__)


class EventSink(Protocol):
    async def emit(self, event: Event) -> None: ...


class RepositorySink:
    def __init__(self, repository: EventRepository) -> None:
        self._repository = repository

    async def emit(self, event: Event) -> None:
        await self._repository.add(event)


class BusSink:
    def __init__(self, bus: Bus, *, qos: int = 1) -> None:
        self._bus = bus
        self._qos = qos

    async def emit(self, event: Event) -> None:
        await self._bus.publish(
            EVENTS_TOPIC, event, params={"camera_id": event.camera_id}, qos=self._qos
        )


@dataclass(slots=True)
class DeliveryStats:
    delivered: int = 0
    failed: int = 0
    dropped: int = 0


class _Worker:
    def __init__(
        self, name: str, sink: EventSink, queue_size: int, attempts: int, retry_delay: float
    ) -> None:
        self.name = name
        self.stats = DeliveryStats()
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=queue_size)
        self._sink = sink
        self._attempts = attempts
        self._retry_delay = retry_delay

    def offer(self, event: Event) -> None:
        if self.queue.full():
            self.queue.get_nowait()
            self.queue.task_done()
            self.stats.dropped += 1
        self.queue.put_nowait(event)

    async def run(self) -> None:
        while True:
            event = await self.queue.get()
            try:
                await self._deliver(event)
            finally:
                self.queue.task_done()

    async def _deliver(self, event: Event) -> None:
        for attempt in range(1, self._attempts + 1):
            try:
                await self._sink.emit(event)
            except Exception as exc:
                if attempt == self._attempts:
                    self.stats.failed += 1
                    logger.error("sink %s gave up on event %s: %s", self.name, event.id, exc)
                    return
                await asyncio.sleep(self._retry_delay * 2 ** (attempt - 1))
            else:
                self.stats.delivered += 1
                return


class FanOut:
    def __init__(
        self,
        sinks: Mapping[str, EventSink],
        *,
        queue_size: int = 1000,
        attempts: int = 3,
        retry_delay: float = 0.5,
        drain_timeout: float = 5.0,
    ) -> None:
        if queue_size < 1 or attempts < 1:
            raise ValueError("queue_size and attempts must be at least 1")
        self._workers = [
            _Worker(name, sink, queue_size, attempts, retry_delay) for name, sink in sinks.items()
        ]
        self._drain_timeout = drain_timeout
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def stats(self) -> dict[str, DeliveryStats]:
        return {worker.name: worker.stats for worker in self._workers}

    async def __aenter__(self) -> FanOut:
        self._tasks = [
            asyncio.create_task(worker.run(), name=f"sink-{worker.name}")
            for worker in self._workers
        ]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        drained = asyncio.gather(*(worker.queue.join() for worker in self._workers))
        try:
            await asyncio.wait_for(drained, self._drain_timeout)
        except TimeoutError:
            pending = sum(worker.queue.qsize() for worker in self._workers)
            logger.warning("shutting down with %d undelivered events", pending)
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def emit(self, event: Event) -> None:
        for worker in self._workers:
            worker.offer(event)
