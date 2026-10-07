from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class Side(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    ON = "on"


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"bottom-right corner precedes top-left corner: {self}")

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def anchor(self) -> Point:
        return Point((self.x1 + self.x2) / 2, self.y2)

    def iou(self, other: BBox) -> float:
        width = min(self.x2, other.x2) - max(self.x1, other.x1)
        height = min(self.y2, other.y2) - max(self.y1, other.y1)
        if width <= 0 or height <= 0:
            return 0.0
        overlap = width * height
        return overlap / (self.area + other.area - overlap)

    def clip(self, width: float, height: float) -> BBox:
        return BBox(
            min(max(self.x1, 0), width),
            min(max(self.y1, 0), height),
            min(max(self.x2, 0), width),
            min(max(self.y2, 0), height),
        )

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True, slots=True)
class Polygon:
    vertices: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("a polygon needs at least three vertices")

    @classmethod
    def of(cls, coordinates: Iterable[tuple[float, float]]) -> Polygon:
        return cls(tuple(Point(x, y) for x, y in coordinates))

    def contains(self, point: Point) -> bool:
        inside = False
        previous = self.vertices[-1]
        for current in self.vertices:
            if (current.y > point.y) != (previous.y > point.y):
                t = (point.y - current.y) / (previous.y - current.y)
                if point.x < current.x + t * (previous.x - current.x):
                    inside = not inside
            previous = current
        return inside


@dataclass(frozen=True, slots=True)
class Line:
    start: Point
    end: Point

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("a line needs a non-zero length")

    def side(self, point: Point) -> Side:
        value = _cross(self.start, self.end, point)
        if value > 0:
            return Side.RIGHT
        if value < 0:
            return Side.LEFT
        return Side.ON

    def crossing(self, before: Point, after: Point) -> Side | None:
        origin, target = self.side(before), self.side(after)
        if Side.ON in (origin, target) or origin is target:
            return None
        if _cross(before, after, self.start) * _cross(before, after, self.end) > 0:
            return None
        return target


def _cross(origin: Point, towards: Point, point: Point) -> float:
    return (towards.x - origin.x) * (point.y - origin.y) - (towards.y - origin.y) * (
        point.x - origin.x
    )
