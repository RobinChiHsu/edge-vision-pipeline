from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from .geometry import Line, Polygon, Side
from .tracking import Track


@dataclass(frozen=True, slots=True)
class RuleHit:
    rule_id: str
    kind: str
    track: Track
    details: Mapping[str, Any] = field(default_factory=dict)


class Rule(Protocol):
    @property
    def id(self) -> str: ...

    def evaluate(self, tracks: Sequence[Track], now: float) -> list[RuleHit]: ...


class Direction(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class _LabelFilter:
    def __init__(self, labels: Collection[str] | None) -> None:
        self._labels = frozenset(labels) if labels is not None else None

    def accepts(self, track: Track) -> bool:
        return self._labels is None or track.label in self._labels


class ZoneRule:
    def __init__(
        self,
        rule_id: str,
        zone: Polygon,
        *,
        labels: Collection[str] | None = None,
        min_dwell: float = 0.0,
    ) -> None:
        if min_dwell < 0:
            raise ValueError("min_dwell must not be negative")
        self.id = rule_id
        self.zone = zone
        self._filter = _LabelFilter(labels)
        self._min_dwell = min_dwell
        self._inside_since: dict[int, float] = {}
        self._reported: set[int] = set()
        self.occupancy = 0

    def evaluate(self, tracks: Sequence[Track], now: float) -> list[RuleHit]:
        self._forget_missing({track.id for track in tracks})
        hits: list[RuleHit] = []
        occupancy = 0
        for track in tracks:
            if not track.visible or not self._filter.accepts(track):
                continue
            if not self.zone.contains(track.anchor):
                self._inside_since.pop(track.id, None)
                self._reported.discard(track.id)
                continue
            occupancy += 1
            hit = self._check_dwell(track, now)
            if hit is not None:
                hits.append(hit)
        self.occupancy = occupancy
        return hits

    def _check_dwell(self, track: Track, now: float) -> RuleHit | None:
        since = self._inside_since.setdefault(track.id, now)
        dwell = now - since
        if track.id in self._reported or dwell < self._min_dwell:
            return None
        self._reported.add(track.id)
        return RuleHit(self.id, "zone_entered", track, {"dwell": round(dwell, 3)})

    def _forget_missing(self, alive: set[int]) -> None:
        for track_id in set(self._inside_since) - alive:
            del self._inside_since[track_id]
        self._reported &= alive


class LineRule:
    def __init__(
        self,
        rule_id: str,
        line: Line,
        *,
        direction: Direction = Direction.BOTH,
        labels: Collection[str] | None = None,
        cooldown: float = 1.0,
    ) -> None:
        if cooldown < 0:
            raise ValueError("cooldown must not be negative")
        self.id = rule_id
        self.line = line
        self._direction = direction
        self._filter = _LabelFilter(labels)
        self._cooldown = cooldown
        self._last_crossing: dict[int, float] = {}
        self._counts = {Side.LEFT: 0, Side.RIGHT: 0}

    @property
    def counts(self) -> dict[str, int]:
        return {side.value: count for side, count in self._counts.items()}

    def evaluate(self, tracks: Sequence[Track], now: float) -> list[RuleHit]:
        alive = {track.id for track in tracks}
        self._last_crossing = {k: v for k, v in self._last_crossing.items() if k in alive}
        hits: list[RuleHit] = []
        for track in tracks:
            side = self._crossing(track)
            if side is None or self._cooling_down(track.id, now):
                continue
            self._last_crossing[track.id] = now
            if self._direction in (Direction.BOTH, side.value):
                self._counts[side] += 1
                details = {"direction": side.value, "count": self._counts[side]}
                hits.append(RuleHit(self.id, "line_crossed", track, details))
        return hits

    def _crossing(self, track: Track) -> Side | None:
        if not track.visible or track.previous_anchor is None or not self._filter.accepts(track):
            return None
        return self.line.crossing(track.previous_anchor, track.anchor)

    def _cooling_down(self, track_id: int, now: float) -> bool:
        last = self._last_crossing.get(track_id)
        return last is not None and now - last < self._cooldown
