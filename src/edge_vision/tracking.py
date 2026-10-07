from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from .detectors import Detection
from .geometry import BBox, Point

VELOCITY_SMOOTHING = 0.5


@dataclass(frozen=True, slots=True)
class Track:
    id: int
    label: str
    bbox: BBox
    confidence: float
    first_seen: float
    last_seen: float
    hits: int
    missed: int
    previous_anchor: Point | None

    @property
    def anchor(self) -> Point:
        return self.bbox.anchor

    @property
    def visible(self) -> bool:
        return self.missed == 0


class _TrackState:
    def __init__(self, track_id: int, detection: Detection, timestamp: float) -> None:
        self.id = track_id
        self.label = detection.label
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.first_seen = self.last_seen = timestamp
        self.hits = 1
        self.missed = 0
        self.previous_anchor: Point | None = None
        self.velocity: tuple[float, float] | None = None

    def predicted(self, timestamp: float) -> BBox:
        if self.velocity is None:
            return self.bbox
        elapsed = timestamp - self.last_seen
        dx, dy = self.velocity[0] * elapsed, self.velocity[1] * elapsed
        box = self.bbox
        return BBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy)

    def observe(self, detection: Detection, timestamp: float) -> None:
        elapsed = timestamp - self.last_seen
        if elapsed > 0:
            old, new = self.bbox.anchor, detection.bbox.anchor
            measured = ((new.x - old.x) / elapsed, (new.y - old.y) / elapsed)
            self.velocity = measured if self.velocity is None else _blend(self.velocity, measured)
        self.previous_anchor = self.bbox.anchor
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.last_seen = timestamp
        self.hits += 1
        self.missed = 0

    def snapshot(self) -> Track:
        return Track(
            id=self.id,
            label=self.label,
            bbox=self.bbox,
            confidence=self.confidence,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            hits=self.hits,
            missed=self.missed,
            previous_anchor=self.previous_anchor,
        )


class IouTracker:
    def __init__(
        self, *, iou_threshold: float = 0.3, max_missed: int = 15, min_hits: int = 3
    ) -> None:
        if not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must be within (0, 1]")
        if max_missed < 0:
            raise ValueError("max_missed must not be negative")
        if min_hits < 1:
            raise ValueError("min_hits must be at least 1")
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed
        self._min_hits = min_hits
        self._tracks: list[_TrackState] = []
        self._ids = itertools.count(1)

    def update(self, detections: Sequence[Detection], timestamp: float) -> list[Track]:
        matches = self._match(detections, timestamp)
        for track_index, detection_index in matches.items():
            self._tracks[track_index].observe(detections[detection_index], timestamp)
        for index, track in enumerate(self._tracks):
            if index not in matches:
                track.missed += 1
        self._tracks = [track for track in self._tracks if self._alive(track)]
        matched = set(matches.values())
        for index, detection in enumerate(detections):
            if index not in matched:
                self._tracks.append(_TrackState(next(self._ids), detection, timestamp))
        return [track.snapshot() for track in self._tracks if track.hits >= self._min_hits]

    def _match(self, detections: Sequence[Detection], timestamp: float) -> dict[int, int]:
        candidates = sorted(
            (
                (track.predicted(timestamp).iou(detection.bbox), t, d)
                for t, track in enumerate(self._tracks)
                for d, detection in enumerate(detections)
                if track.label == detection.label
            ),
            reverse=True,
        )
        matches: dict[int, int] = {}
        used: set[int] = set()
        for iou, t, d in candidates:
            if iou < self._iou_threshold:
                break
            if t not in matches and d not in used:
                matches[t] = d
                used.add(d)
        return matches

    def _alive(self, track: _TrackState) -> bool:
        if track.hits < self._min_hits:
            return track.missed == 0
        return track.missed <= self._max_missed


def _blend(previous: tuple[float, float], measured: tuple[float, float]) -> tuple[float, float]:
    return (
        previous[0] + VELOCITY_SMOOTHING * (measured[0] - previous[0]),
        previous[1] + VELOCITY_SMOOTHING * (measured[1] - previous[1]),
    )
