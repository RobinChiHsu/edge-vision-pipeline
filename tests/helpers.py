from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
from rtsp_supervisor import SourceError

from edge_vision.detectors import Detection, Frame
from edge_vision.events import Event
from edge_vision.geometry import BBox

BASE_TIME = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


async def wait_until(predicate: Callable[[], bool], within: float = 3.0) -> None:
    async with asyncio.timeout(within):
        while not predicate():
            await asyncio.sleep(0.01)


def make_event(offset: int = 0, **overrides: object) -> Event:
    fields: dict[str, object] = {
        "camera_id": "gate",
        "rule_id": "entrance",
        "kind": "line_crossed",
        "track_id": 1,
        "label": "person",
        "confidence": 0.9,
        "bbox": (0.0, 0.0, 10.0, 10.0),
        "occurred_at": BASE_TIME + timedelta(seconds=offset),
        "details": {},
    }
    return Event.model_validate({**fields, **overrides})


def person_at(x: float, y: float = 100, size: float = 40) -> Detection:
    return Detection("person", 0.9, BBox(x, y, x + size, y + size))


class ScriptedDetector:
    def __init__(self, script: list[list[Detection]]) -> None:
        self._script = list(script)
        self.calls = 0

    def detect(self, frame: Frame) -> list[Detection]:
        self.calls += 1
        return self._script.pop(0) if self._script else []


class MovingSquareSource:
    def __init__(
        self, *, width: int = 200, height: int = 120, step: int = 6, fps: float = 60
    ) -> None:
        self._width, self._height, self._step = width, height, step
        self._interval = 1 / fps
        self._index = 0

    def open(self) -> None:
        self._index = 0

    def read(self) -> Frame:
        time.sleep(self._interval)
        frame = np.full((self._height, self._width, 3), 30, dtype=np.uint8)
        warmup = 15
        x = (self._index - warmup) * self._step
        self._index += 1
        if x > self._width:
            raise SourceError("end of clip")
        if x >= 0:
            frame[40:80, x : x + 30] = 220
        return frame

    def close(self) -> None:
        pass
