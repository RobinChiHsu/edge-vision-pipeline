import pytest

from edge_vision.geometry import BBox, Line, Point, Polygon
from edge_vision.rules import Direction, LineRule, ZoneRule
from edge_vision.tracking import Track

ZONE = Polygon.of([(100, 100), (200, 100), (200, 200), (100, 200)])
VERTICAL = Line(Point(100, 0), Point(100, 300))


def track(
    track_id: int = 1,
    at: tuple[float, float] = (0, 0),
    *,
    came_from: tuple[float, float] | None = None,
    label: str = "person",
    missed: int = 0,
) -> Track:
    x, y = at
    return Track(
        id=track_id,
        label=label,
        bbox=BBox(x - 10, y - 40, x + 10, y),
        confidence=0.9,
        first_seen=0.0,
        last_seen=0.0,
        hits=5,
        missed=missed,
        previous_anchor=Point(*came_from) if came_from else None,
    )


class TestZoneRule:
    def test_reports_entry_once_per_visit(self) -> None:
        rule = ZoneRule("restricted", ZONE)

        first = rule.evaluate([track(at=(150, 150))], now=1.0)
        again = rule.evaluate([track(at=(160, 150))], now=2.0)

        assert [(hit.rule_id, hit.kind, hit.track.id) for hit in first] == [
            ("restricted", "zone_entered", 1)
        ]
        assert again == []

    def test_reports_again_after_leaving_and_returning(self) -> None:
        rule = ZoneRule("restricted", ZONE)
        positions = [(150, 150), (300, 150), (150, 150)]

        hits = [rule.evaluate([track(at=p)], now=float(i)) for i, p in enumerate(positions)]

        assert [len(h) for h in hits] == [1, 0, 1]

    def test_min_dwell_delays_report_until_object_stays(self) -> None:
        rule = ZoneRule("loitering", ZONE, min_dwell=5.0)

        hits = [rule.evaluate([track(at=(150, 150))], now=t) for t in (10.0, 12.0, 15.0, 16.0)]

        assert [len(h) for h in hits] == [0, 0, 1, 0]
        assert hits[2][0].details == {"dwell": 5.0}

    def test_short_occlusion_does_not_reset_dwell(self) -> None:
        rule = ZoneRule("loitering", ZONE, min_dwell=3.0)
        rule.evaluate([track(at=(150, 150))], now=0.0)
        rule.evaluate([track(at=(150, 150), missed=1)], now=2.0)

        assert len(rule.evaluate([track(at=(150, 150))], now=3.0)) == 1

    def test_ignores_other_labels(self) -> None:
        rule = ZoneRule("vehicles", ZONE, labels={"car"})

        assert rule.evaluate([track(at=(150, 150), label="person")], now=0.0) == []

    def test_occupancy_counts_visible_tracks_inside(self) -> None:
        rule = ZoneRule("restricted", ZONE)

        rule.evaluate([track(1, (150, 150)), track(2, (120, 180)), track(3, (500, 500))], now=0.0)
        assert rule.occupancy == 2

        rule.evaluate([track(1, (150, 150))], now=1.0)
        assert rule.occupancy == 1

    def test_forgets_tracks_that_ended(self) -> None:
        rule = ZoneRule("loitering", ZONE, min_dwell=5.0)
        rule.evaluate([track(1, (150, 150))], now=0.0)
        rule.evaluate([], now=1.0)

        assert rule.evaluate([track(1, (150, 150))], now=5.0) == []


class TestLineRule:
    def test_counts_crossing_with_direction(self) -> None:
        rule = LineRule("door", VERTICAL)

        [hit] = rule.evaluate([track(at=(90, 150), came_from=(110, 150))], now=0.0)

        assert hit.kind == "line_crossed"
        assert hit.details == {"direction": "right", "count": 1}
        assert rule.counts == {"left": 0, "right": 1}

    def test_direction_filter(self) -> None:
        rule = LineRule("entry", VERTICAL, direction=Direction.LEFT)

        assert rule.evaluate([track(at=(90, 150), came_from=(110, 150))], now=0.0) == []
        assert len(rule.evaluate([track(2, (110, 150), came_from=(90, 150))], now=0.0)) == 1
        assert rule.counts == {"left": 1, "right": 0}

    def test_cooldown_suppresses_jitter_around_the_line(self) -> None:
        rule = LineRule("door", VERTICAL, cooldown=1.0)
        moves = [((110, 150), (90, 150)), ((90, 150), (110, 150)), ((110, 150), (90, 150))]

        hits = [
            rule.evaluate([track(at=after, came_from=before)], now=t)
            for t, (before, after) in zip((0.0, 0.3, 1.5), moves, strict=True)
        ]

        assert [len(h) for h in hits] == [1, 0, 1]

    def test_cooldown_is_tracked_per_object(self) -> None:
        rule = LineRule("door", VERTICAL, cooldown=10.0)
        tracks = [track(i, (90, 150), came_from=(110, 150)) for i in (1, 2)]

        hits = rule.evaluate(tracks, now=0.0)

        assert len(hits) == 2

    def test_ignores_tracks_without_history_hidden_tracks_and_other_labels(self) -> None:
        rule = LineRule("door", VERTICAL, labels={"person"})

        hits = rule.evaluate(
            [
                track(1, (90, 150)),
                track(2, (90, 150), came_from=(110, 150), missed=2),
                track(3, (90, 150), came_from=(110, 150), label="dog"),
            ],
            now=0.0,
        )

        assert hits == []

    def test_rejects_negative_cooldown(self) -> None:
        with pytest.raises(ValueError, match="cooldown"):
            LineRule("door", VERTICAL, cooldown=-1)


def test_zone_rejects_negative_dwell() -> None:
    with pytest.raises(ValueError, match="min_dwell"):
        ZoneRule("zone", ZONE, min_dwell=-1)
