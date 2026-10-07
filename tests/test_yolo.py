from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from edge_vision.detectors.yolo import OnnxYoloDetector, decode, letterbox, nms
from edge_vision.geometry import BBox


def candidates(*rows: tuple[float, float, float, float, float, float]) -> NDArray[np.float32]:
    return np.array(rows, dtype=np.float32).T[np.newaxis]


class TestLetterbox:
    def test_scales_and_pads_to_square_input(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        tensor, transform = letterbox(frame, 320)

        assert tensor.shape == (1, 3, 320, 320)
        assert tensor.dtype == np.float32
        assert transform.scale == 0.5
        assert (transform.pad_x, transform.pad_y) == (0, 40)

    def test_padding_uses_neutral_grey_and_pixels_are_normalised(self) -> None:
        frame = np.full((100, 200, 3), 255, dtype=np.uint8)

        tensor, _ = letterbox(frame, 200)

        assert tensor[0, :, 0, 0] == pytest.approx([114 / 255] * 3)
        assert tensor[0, :, 100, 100] == pytest.approx([1.0] * 3)

    def test_converts_bgr_to_rgb(self) -> None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[..., 0] = 255

        tensor, _ = letterbox(frame, 10)

        assert tensor[0, 2, 5, 5] == 1.0
        assert tensor[0, 0, 5, 5] == 0.0

    def test_maps_boxes_back_to_original_frame(self) -> None:
        _, transform = letterbox(np.zeros((480, 640, 3), dtype=np.uint8), 320)

        restored = transform.restore(BBox(10, 50, 110, 140))

        assert restored == BBox(20, 20, 220, 200)

    def test_restored_boxes_are_clipped_to_frame(self) -> None:
        _, transform = letterbox(np.zeros((480, 640, 3), dtype=np.uint8), 320)

        assert transform.restore(BBox(0, 0, 320, 320)) == BBox(0, 0, 640, 480)


class TestDecode:
    def test_converts_centre_format_and_picks_best_class(self) -> None:
        output = candidates((50, 50, 20, 10, 0.1, 0.9), (10, 10, 4, 4, 0.05, 0.1))

        boxes, scores, classes = decode(output, confidence=0.5)

        assert boxes.tolist() == [[40, 45, 60, 55]]
        assert scores.tolist() == pytest.approx([0.9])
        assert classes.tolist() == [1]

    def test_accepts_output_without_batch_dimension(self) -> None:
        boxes, _, _ = decode(candidates((5, 5, 2, 2, 0.8, 0.0))[0], confidence=0.5)

        assert boxes.shape == (1, 4)


class TestNms:
    def test_suppresses_overlapping_lower_score_box(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
        scores = np.array([0.6, 0.9, 0.5], dtype=np.float32)

        assert nms(boxes, scores, iou_threshold=0.5) == [1, 2]

    def test_empty_input(self) -> None:
        assert nms(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.5) == []

    @given(
        corners=arrays(
            np.float32, (12, 2), elements=st.floats(0, 100, width=32, allow_subnormal=False)
        ),
        sizes=arrays(
            np.float32, (12, 2), elements=st.floats(1, 40, width=32, allow_subnormal=False)
        ),
        scores=arrays(
            np.float32,
            12,
            elements=st.floats(0.0625, 1, width=32, allow_subnormal=False),
            unique=True,
        ),
        threshold=st.floats(0.1, 0.9),
    )
    def test_kept_boxes_never_overlap_beyond_threshold(
        self,
        corners: NDArray[np.float32],
        sizes: NDArray[np.float32],
        scores: NDArray[np.float32],
        threshold: float,
    ) -> None:
        boxes = np.concatenate([corners, corners + sizes], axis=1)

        kept = nms(boxes, scores, threshold)

        assert kept[0] == int(scores.argmax())
        as_boxes = [BBox(*boxes[i].tolist()) for i in kept]
        for i, first in enumerate(as_boxes):
            for second in as_boxes[i + 1 :]:
                assert first.iou(second) <= threshold + 1e-5


class FakeInput:
    name = "images"


class FakeSession:
    def __init__(self, output: NDArray[np.float32]) -> None:
        self.output = output
        self.feeds: list[Mapping[str, Any]] = []

    def get_inputs(self) -> Sequence[FakeInput]:
        return [FakeInput()]

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any]
    ) -> list[NDArray[np.float32]]:
        self.feeds.append(input_feed)
        return [self.output]


class TestOnnxYoloDetector:
    def test_runs_model_and_returns_detections_in_frame_coordinates(self) -> None:
        output = candidates(
            (160, 160, 40, 40, 0.92, 0.01),
            (162, 160, 40, 40, 0.80, 0.02),
            (60, 200, 20, 60, 0.05, 0.70),
            (300, 300, 10, 10, 0.10, 0.10),
        )
        session = FakeSession(output)
        detector = OnnxYoloDetector(
            "model.onnx", ["person", "car"], input_size=320, session_factory=lambda _: session
        )

        detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        assert [(d.label, round(d.confidence, 2)) for d in detections] == [
            ("person", 0.92),
            ("car", 0.70),
        ]
        assert detections[0].bbox == BBox(280, 200, 360, 280)
        assert session.feeds[0]["images"].shape == (1, 3, 320, 320)

    def test_unknown_class_ids_get_a_generic_label(self) -> None:
        session = FakeSession(candidates((10, 10, 4, 4, 0.0, 0.9)))
        detector = OnnxYoloDetector(
            "m.onnx", ["person"], input_size=32, session_factory=lambda _: session
        )

        [detection] = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

        assert detection.label == "class_1"

    def test_class_filter_drops_other_labels(self) -> None:
        output = candidates((10, 10, 4, 4, 0.9, 0.0), (20, 20, 4, 4, 0.0, 0.9))
        detector = OnnxYoloDetector(
            "m.onnx",
            ["person", "car"],
            input_size=32,
            classes={"car"},
            session_factory=lambda _: FakeSession(output),
        )

        assert [d.label for d in detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))] == ["car"]

    def test_default_session_requires_onnxruntime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(__import__("sys").modules, "onnxruntime", None)

        with pytest.raises(ImportError, match="onnx"):
            OnnxYoloDetector("model.onnx", ["person"])


def test_default_session_uses_available_onnxruntime_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[str, list[str]]] = []

    class FakeRuntime:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(path: str, providers: list[str]) -> FakeSession:
            created.append((path, providers))
            return FakeSession(candidates((1, 1, 1, 1, 0.0, 0.0)))

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeRuntime)

    OnnxYoloDetector("model.onnx", ["person"])

    assert created == [("model.onnx", ["CPUExecutionProvider"])]
