from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import numpy as np
import pytest
from rtsp_supervisor import ExponentialBackoff, StreamSupervisor, SupervisorConfig
from rtsp_supervisor import Frame as StreamFrame

from edge_vision.detectors import Detection, Frame
from edge_vision.detectors.motion import MotionDetector
from edge_vision.events import Event
from edge_vision.geometry import Line, Point
from edge_vision.pipeline import CameraPipeline
from edge_vision.rules import LineRule
from edge_vision.tracking import IouTracker
from tests.helpers import MovingSquareSource, ScriptedDetector, person_at, wait_until

BLANK: Frame = np.zeros((10, 10, 3), dtype=np.uint8)
NOW = datetime(2026, 3, 1, 9, 30, tzinfo=UTC)
DOOR = Line(Point(100, 0), Point(100, 300))


class CollectingSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class ListFeed:
    def __init__(self, timestamps: list[float]) -> None:
        self._timestamps = timestamps

    async def frames(self) -> AsyncIterator[StreamFrame[Frame]]:
        for index, timestamp in enumerate(self._timestamps):
            yield StreamFrame(data=BLANK, sequence=index + 1, captured_at=timestamp)


class BrokenDetector:
    def detect(self, frame: Frame) -> list[Detection]:
        raise RuntimeError("model crashed")


def make_pipeline(
    detector: ScriptedDetector | BrokenDetector | MotionDetector,
    sink: CollectingSink,
    **options: float,
) -> CameraPipeline:
    return CameraPipeline(
        "gate",
        detector,
        IouTracker(min_hits=1),
        [LineRule("door", DOOR)],
        sink,
        wall_clock=lambda: NOW,
        **options,
    )


async def test_turns_line_crossing_into_event() -> None:
    sink = CollectingSink()
    detector = ScriptedDetector([[person_at(x)] for x in (40, 60, 80, 100)])
    pipeline = make_pipeline(detector, sink)

    for index in range(4):
        await pipeline.process(BLANK, timestamp=index * 0.1)

    [event] = sink.events
    assert (event.camera_id, event.rule_id, event.kind) == ("gate", "door", "line_crossed")
    assert (event.track_id, event.label) == (1, "person")
    assert event.occurred_at == NOW
    assert event.details == {"direction": "left", "count": 1}
    assert pipeline.stats.events_emitted == 1
    assert pipeline.stats.frames_processed == 4
    assert pipeline.stats.active_tracks == 1
    assert pipeline.stats.inference_ms is not None


async def test_max_fps_skips_frames_that_arrive_too_soon() -> None:
    detector = ScriptedDetector([])
    pipeline = make_pipeline(detector, CollectingSink(), max_fps=5)

    await pipeline.run(ListFeed([0.0, 0.05, 0.1, 0.21, 0.3, 0.45]))

    assert detector.calls == 3
    assert pipeline.stats.frames_skipped == 3


async def test_detector_failure_is_counted_and_pipeline_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = make_pipeline(BrokenDetector(), CollectingSink())

    await pipeline.run(ListFeed([0.0, 0.1]))

    assert pipeline.stats.detector_errors == 2
    assert pipeline.stats.frames_processed == 0
    assert "model crashed" in caplog.text


def test_rejects_non_positive_max_fps() -> None:
    with pytest.raises(ValueError, match="max_fps"):
        make_pipeline(ScriptedDetector([]), CollectingSink(), max_fps=0)


async def test_end_to_end_with_stream_supervisor_and_motion_detector() -> None:
    sink = CollectingSink()
    supervisor: StreamSupervisor[Frame] = StreamSupervisor(
        MovingSquareSource,
        config=SupervisorConfig(read_timeout=1.0),
        backoff=ExponentialBackoff(initial=10.0, maximum=10.0),
    )
    pipeline = make_pipeline(MotionDetector(min_area=200, warmup_frames=10), sink)

    async with supervisor:
        runner = asyncio.create_task(pipeline.run(supervisor))
        await wait_until(lambda: len(sink.events) >= 1, within=5.0)
        runner.cancel()

    [event] = sink.events[:1]
    assert event.kind == "line_crossed"
    assert event.label == "motion"
    assert event.details["direction"] == "left"
