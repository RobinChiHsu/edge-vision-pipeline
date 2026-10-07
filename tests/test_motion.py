import numpy as np
import pytest

from edge_vision.detectors.base import Frame
from edge_vision.detectors.motion import MotionDetector

WIDTH, HEIGHT = 160, 120


def scene(square_at: int | None = None, size: int = 20) -> Frame:
    frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    if square_at is not None:
        frame[50 : 50 + size, square_at : square_at + size] = 230
    return frame


def warmed_up(detector: MotionDetector, frames: int = 30) -> MotionDetector:
    for _ in range(frames):
        detector.detect(scene())
    return detector


def test_static_scene_produces_no_detections() -> None:
    detector = warmed_up(MotionDetector(min_area=50))

    assert detector.detect(scene()) == []


def test_moving_object_is_boxed_tightly() -> None:
    detector = warmed_up(MotionDetector(min_area=50))

    [detection] = detector.detect(scene(square_at=100))

    box = detection.bbox
    assert detection.label == "motion"
    assert box.x1 <= 100
    assert box.y1 <= 50
    assert box.x2 >= 120
    assert box.y2 >= 70
    assert box.x2 - box.x1 <= 30
    assert box.y2 - box.y1 <= 30


def test_blobs_smaller_than_min_area_are_ignored() -> None:
    detector = warmed_up(MotionDetector(min_area=500))

    assert detector.detect(scene(square_at=100, size=10)) == []


def test_frames_during_warmup_are_ignored() -> None:
    detector = MotionDetector(min_area=50, warmup_frames=5)

    assert detector.detect(scene(square_at=100)) == []


def test_confidence_grows_with_blob_size() -> None:
    small = warmed_up(MotionDetector(min_area=50)).detect(scene(square_at=20, size=10))
    large = warmed_up(MotionDetector(min_area=50)).detect(scene(square_at=20, size=40))

    assert 0 < small[0].confidence < large[0].confidence <= 1


@pytest.mark.parametrize("kwargs", [{"min_area": 0}, {"warmup_frames": -1}])
def test_rejects_invalid_settings(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        MotionDetector(**kwargs)
