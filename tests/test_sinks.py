from __future__ import annotations

import asyncio
import logging

import pytest
from typed_mqtt_bus import Bus, InMemoryBroker

from edge_vision.events import Event
from edge_vision.sinks import EVENTS_TOPIC, BusSink, FanOut, RepositorySink
from edge_vision.storage import EventPage, EventQuery
from tests.helpers import make_event, wait_until


class RecordingSink:
    def __init__(self, failures: int = 0, delay: float = 0.0) -> None:
        self.received: list[Event] = []
        self._failures = failures
        self._delay = delay

    async def emit(self, event: Event) -> None:
        await asyncio.sleep(self._delay)
        if self._failures > 0:
            self._failures -= 1
            raise ConnectionError("sink unavailable")
        self.received.append(event)


class MemoryRepository:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def add(self, event: Event) -> None:
        self.events.append(event)

    async def get(self, event_id: object) -> Event | None:
        return None

    async def list(self, query: EventQuery) -> EventPage:
        return EventPage(items=self.events, next_cursor=None)

    async def ping(self) -> None:
        pass


async def test_every_sink_receives_every_event() -> None:
    first, second = RecordingSink(), RecordingSink()

    async with FanOut({"first": first, "second": second}) as fan_out:
        for offset in range(3):
            await fan_out.emit(make_event(offset))

    assert len(first.received) == len(second.received) == 3
    assert fan_out.stats["first"].delivered == 3


async def test_slow_sink_does_not_delay_fast_sink() -> None:
    slow, fast = RecordingSink(delay=0.5), RecordingSink()

    async with FanOut({"slow": slow, "fast": fast}, drain_timeout=0.05) as fan_out:
        await fan_out.emit(make_event())
        await wait_until(lambda: len(fast.received) == 1, within=0.2)

    assert slow.received == []


async def test_transient_failure_is_retried() -> None:
    flaky = RecordingSink(failures=2)

    async with FanOut({"db": flaky}, retry_delay=0.001) as fan_out:
        await fan_out.emit(make_event())

    assert len(flaky.received) == 1
    assert fan_out.stats["db"].failed == 0


async def test_persistent_failure_is_counted_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    broken = RecordingSink(failures=10)

    async with FanOut({"db": broken}, attempts=2, retry_delay=0.001) as fan_out:
        await fan_out.emit(make_event())

    assert fan_out.stats["db"].failed == 1
    assert "sink db gave up" in caplog.text


async def test_full_queue_drops_oldest_event() -> None:
    blocked = RecordingSink(delay=0.2)

    async with FanOut({"slow": blocked}, queue_size=2) as fan_out:
        events = [make_event(offset) for offset in range(5)]
        for event in events:
            await fan_out.emit(event)

    assert fan_out.stats["slow"].dropped >= 2
    assert blocked.received[-1] == events[-1]


async def test_shutdown_gives_up_after_drain_timeout(caplog: pytest.LogCaptureFixture) -> None:
    stuck = RecordingSink(delay=10)

    async with FanOut({"stuck": stuck}, drain_timeout=0.05) as fan_out:
        await fan_out.emit(make_event())
        await fan_out.emit(make_event(1))

    assert "undelivered" in caplog.text


def test_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="queue_size"):
        FanOut({}, queue_size=0)


async def test_repository_sink_stores_event() -> None:
    repository = MemoryRepository()

    await RepositorySink(repository).emit(make_event())

    assert len(repository.events) == 1


async def test_bus_sink_publishes_on_camera_topic() -> None:
    broker = InMemoryBroker()
    async with Bus(broker.client()) as bus, bus.subscribe(EVENTS_TOPIC) as events:
        await BusSink(bus).emit(make_event(camera_id="dock"))
        envelope = await asyncio.wait_for(anext(events), 1)

    assert envelope.params == {"camera_id": "dock"}
    assert broker.history[0].topic == "edge-vision/cameras/dock/events"
    assert broker.history[0].qos == 1


async def test_logging_is_quiet_on_success(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        async with FanOut({"ok": RecordingSink()}) as fan_out:
            await fan_out.emit(make_event())

    assert caplog.text == ""
