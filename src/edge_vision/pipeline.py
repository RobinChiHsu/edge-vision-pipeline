from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from rtsp_supervisor import Frame as StreamFrame

from .detectors import Detector, Frame
from .events import Event
from .rules import Rule
from .sinks import EventSink
from .tracking import IouTracker

LATENCY_SMOOTHING = 0.2

logger = logging.getLogger(__name__)


class FrameFeed(Protocol):
    def frames(self) -> AsyncIterator[StreamFrame[Frame]]: ...


@dataclass(slots=True)
class PipelineStats:
    frames_processed: int = 0
    frames_skipped: int = 0
    detector_errors: int = 0
    events_emitted: int = 0
    active_tracks: int = 0
    inference_ms: float | None = None


class CameraPipeline:
    def __init__(
        self,
        camera_id: str,
        detector: Detector,
        tracker: IouTracker,
        rules: Sequence[Rule],
        sink: EventSink,
        *,
        max_fps: float | None = None,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if max_fps is not None and max_fps <= 0:
            raise ValueError("max_fps must be positive")
        self.camera_id = camera_id
        self.rules = tuple(rules)
        self.stats = PipelineStats()
        self._detector = detector
        self._tracker = tracker
        self._sink = sink
        self._min_interval = 1 / max_fps if max_fps else 0.0
        self._wall_clock = wall_clock
        self._last_processed: float | None = None

    async def run(self, feed: FrameFeed) -> None:
        async for frame in feed.frames():
            if self._throttled(frame.captured_at):
                self.stats.frames_skipped += 1
                continue
            await self.process(frame.data, frame.captured_at)

    async def process(self, image: Frame, timestamp: float) -> list[Event]:
        self._last_processed = timestamp
        started = time.perf_counter()
        try:
            detections = await asyncio.to_thread(self._detector.detect, image)
        except Exception:
            self.stats.detector_errors += 1
            logger.exception("detector failed on camera %s", self.camera_id)
            return []
        self._record_latency((time.perf_counter() - started) * 1000)
        tracks = self._tracker.update(detections, timestamp)
        occurred_at = self._wall_clock()
        events = [
            Event.from_hit(self.camera_id, hit, occurred_at)
            for rule in self.rules
            for hit in rule.evaluate(tracks, timestamp)
        ]
        for event in events:
            await self._sink.emit(event)
        self.stats.frames_processed += 1
        self.stats.events_emitted += len(events)
        self.stats.active_tracks = sum(1 for track in tracks if track.visible)
        return events

    def _throttled(self, timestamp: float) -> bool:
        last = self._last_processed
        return last is not None and timestamp - last < self._min_interval

    def _record_latency(self, milliseconds: float) -> None:
        previous = self.stats.inference_ms
        self.stats.inference_ms = (
            milliseconds
            if previous is None
            else previous + LATENCY_SMOOTHING * (milliseconds - previous)
        )
