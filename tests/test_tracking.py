from itertools import pairwise

import pytest

from edge_vision.detectors import Detection
from edge_vision.geometry import BBox, Point
from edge_vision.tracking import IouTracker, Track


def box_at(x: float, y: float = 100, size: float = 50, label: str = "person") -> Detection:
    return Detection(label, 0.9, BBox(x, y, x + size, y + size))


def run(tracker: IouTracker, frames: list[list[Detection]], fps: float = 10) -> list[list[Track]]:
    return [tracker.update(detections, index / fps) for index, detections in enumerate(frames)]


def ids(tracks: list[Track]) -> list[int]:
    return sorted(track.id for track in tracks)


def test_track_is_confirmed_after_min_hits() -> None:
    history = run(IouTracker(min_hits=3), [[box_at(0)]] * 4)

    assert [len(tracks) for tracks in history] == [0, 0, 1, 1]


def test_stationary_object_keeps_its_id() -> None:
    history = run(IouTracker(min_hits=1), [[box_at(10)]] * 5)

    assert {track.id for tracks in history for track in tracks} == {1}
    assert history[-1][0].hits == 5
    assert history[-1][0].first_seen == 0
    assert history[-1][0].last_seen == pytest.approx(0.4)


ACCELERATING = [0, 15, 40, 75, 120, 165, 210, 255]


def test_accelerating_object_keeps_its_id_through_motion_prediction() -> None:
    history = run(IouTracker(min_hits=1, iou_threshold=0.3), [[box_at(x)] for x in ACCELERATING])

    assert {track.id for tracks in history for track in tracks} == {1}


def test_accelerating_object_outruns_plain_iou_matching() -> None:
    steps = pairwise(ACCELERATING)

    assert min(box_at(a).bbox.iou(box_at(b).bbox) for a, b in steps) < 0.3


def test_objects_with_different_labels_never_match() -> None:
    tracker = IouTracker(min_hits=1)
    tracker.update([box_at(0, label="person")], 0.0)

    tracks = tracker.update([box_at(0, label="car")], 0.1)

    [visible] = [track for track in tracks if track.visible]
    assert (visible.id, visible.label) == (2, "car")


def test_neighbouring_objects_keep_separate_ids() -> None:
    frames = [[box_at(x, y=0), box_at(x, y=60)] for x in range(0, 100, 10)]

    history = run(IouTracker(min_hits=1), frames)

    assert all(ids(tracks) == [1, 2] for tracks in history)
    assert [track.bbox.y1 for track in sorted(history[-1], key=lambda t: t.id)] == [0, 60]


def test_short_occlusion_keeps_id_and_reports_missed_frames() -> None:
    frames = [[box_at(0)], [box_at(5)], [], [], [box_at(20)]]

    history = run(IouTracker(min_hits=1, max_missed=3), frames)

    assert [t.missed for t in history[3]] == [2]
    assert not history[3][0].visible
    assert ids(history[4]) == [1]
    assert history[4][0].visible


def test_long_occlusion_starts_a_new_track() -> None:
    frames = [[box_at(0)], [], [], [box_at(0)]]

    history = run(IouTracker(min_hits=1, max_missed=1), frames)

    assert ids(history[2]) == []
    assert ids(history[3]) == [2]


def test_tentative_track_is_dropped_on_first_miss() -> None:
    frames = [[box_at(0)], [], [box_at(0)], [box_at(0)]]

    history = run(IouTracker(min_hits=2, max_missed=5), frames)

    assert ids(history[3]) == [2]


def test_previous_anchor_follows_the_last_matched_position() -> None:
    tracker = IouTracker(min_hits=1)
    [first] = tracker.update([box_at(0)], 0.0)
    assert first.previous_anchor is None

    [moved] = tracker.update([box_at(10)], 0.1)

    assert moved.previous_anchor == Point(25, 150)
    assert moved.anchor == Point(35, 150)


def test_each_detection_matches_at_most_one_track() -> None:
    tracker = IouTracker(min_hits=1)
    tracker.update([box_at(0), box_at(5)], 0.0)

    tracks = tracker.update([box_at(2)], 0.1)

    assert [t.visible for t in sorted(tracks, key=lambda t: t.id)].count(True) == 1


@pytest.mark.parametrize(
    "kwargs", [{"iou_threshold": 0.0}, {"iou_threshold": 1.5}, {"max_missed": -1}, {"min_hits": 0}]
)
def test_rejects_invalid_settings(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        IouTracker(**kwargs)  # type: ignore[arg-type]
