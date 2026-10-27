import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from edge_vision.detectors.yolo import (
    BatchingSession,
    OnnxYoloDetector,
    SessionPool,
    decode,
    letterbox,
    nms,
)
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

    def __init__(self, shape: Sequence[int | str | None] = (1, 3, 640, 640)) -> None:
        self.shape = shape


class FakeSession:
    def __init__(
        self, output: NDArray[np.float32], shape: Sequence[int | str | None] = (1, 3, 640, 640)
    ) -> None:
        self.output = output
        self.shape = shape
        self.feeds: list[Mapping[str, Any]] = []

    def get_inputs(self) -> Sequence[FakeInput]:
        return [FakeInput(self.shape)]

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


def test_default_session_uses_available_providers_and_limits_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[str, int, list[str]]] = []

    class FakeOptions:
        intra_op_num_threads = 0

    class FakeRuntime:
        SessionOptions = FakeOptions

        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(
            path: str, sess_options: FakeOptions, providers: list[str]
        ) -> FakeSession:
            created.append((path, sess_options.intra_op_num_threads, providers))
            return FakeSession(candidates((1, 1, 1, 1, 0.0, 0.0)))

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeRuntime)

    SessionPool(intra_op_threads=3).get("model.onnx")

    assert created == [("model.onnx", 3, ["CPUExecutionProvider"])]


class TestSessionPool:
    def test_detectors_of_the_same_model_share_one_session(self) -> None:
        created: list[tuple[str, int]] = []

        def factory(path: str, threads: int) -> FakeSession:
            created.append((path, threads))
            return FakeSession(candidates((1, 1, 1, 1, 0.0, 0.0)))

        pool = SessionPool(intra_op_threads=2, session_factory=factory)

        first = OnnxYoloDetector("a.onnx", ["person"], session_factory=pool.get)
        second = OnnxYoloDetector("a.onnx", ["person"], confidence=0.8, session_factory=pool.get)
        OnnxYoloDetector("b.onnx", ["person"], session_factory=pool.get)

        assert first._session is second._session
        assert created == [("a.onnx", 2), ("b.onnx", 2)]

    def test_defaults_to_half_the_cores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("os.cpu_count", lambda: 8)

        assert SessionPool().intra_op_threads == 4

    def test_keeps_one_thread_on_a_single_core(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("os.cpu_count", lambda: 1)

        assert SessionPool().intra_op_threads == 1

    def test_rejects_non_positive_thread_count(self) -> None:
        with pytest.raises(ValueError, match="intra_op_threads"):
            SessionPool(intra_op_threads=0)

    def test_wraps_models_with_a_dynamic_batch_axis(self) -> None:
        pool = SessionPool(
            intra_op_threads=1,
            session_factory=lambda path, _: FakeSession(candidates(), ("batch", 3, 32, 32)),
        )
        try:
            assert isinstance(pool.get("dynamic.onnx"), BatchingSession)
        finally:
            pool.close()

    @pytest.mark.parametrize(
        ("shape", "max_batch"), [((1, 3, 32, 32), 8), (("batch", 3, 32, 32), 1)]
    )
    def test_fixed_batch_models_or_disabled_batching_use_the_plain_session(
        self, shape: tuple[int | str, ...], max_batch: int
    ) -> None:
        session = FakeSession(candidates(), shape)
        pool = SessionPool(
            intra_op_threads=1, max_batch=max_batch, session_factory=lambda *_: session
        )

        assert pool.get("model.onnx") is session


class DoublingSession:
    """Returns each input times two, so every caller can check it got its own slice back."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.gate = threading.Event()
        self.gate.set()
        self.started = threading.Event()
        self.error: Exception | None = None

    def get_inputs(self) -> Sequence[FakeInput]:
        return [FakeInput(("batch", 3, 2, 2))]

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any]
    ) -> list[NDArray[np.float32]]:
        images = input_feed["images"]
        self.batch_sizes.append(len(images))
        self.started.set()
        self.gate.wait(5)
        if self.error is not None:
            raise self.error
        return [images * 2, images.sum(axis=(1, 2, 3))]


def image(value: float, size: int = 2) -> NDArray[np.float32]:
    return np.full((1, 3, size, size), value, dtype=np.float32)


@pytest.fixture
def model() -> DoublingSession:
    return DoublingSession()


@pytest.fixture
def batching(model: DoublingSession) -> Iterator[BatchingSession]:
    session = BatchingSession(model, max_batch=3)
    yield session
    session.close()


def wait_for_queue(session: BatchingSession, size: int) -> None:
    deadline = time.monotonic() + 5
    while session._queue.qsize() < size:
        assert time.monotonic() < deadline, "requests did not queue up"
        time.sleep(0.001)


class TestBatchingSession:
    def test_requests_waiting_for_a_busy_model_run_as_one_batch(
        self, model: DoublingSession, batching: BatchingSession
    ) -> None:
        model.gate.clear()
        with ThreadPoolExecutor(4) as pool:
            first = pool.submit(batching.run, None, {"images": image(1)})
            model.started.wait(5)
            rest = [pool.submit(batching.run, None, {"images": image(v)}) for v in (2, 3)]
            wait_for_queue(batching, 2)
            model.gate.set()

            results = [future.result(5) for future in [first, *rest]]

        assert model.batch_sizes == [1, 2]
        for value, (doubled, total) in zip((1, 2, 3), results, strict=True):
            assert doubled.shape == (1, 3, 2, 2)
            assert np.all(doubled == value * 2)
            assert total.tolist() == [value * 12]

    def test_batches_are_capped_at_max_batch(
        self, model: DoublingSession, batching: BatchingSession
    ) -> None:
        model.gate.clear()
        with ThreadPoolExecutor(5) as pool:
            first = pool.submit(batching.run, None, {"images": image(0)})
            model.started.wait(5)
            rest = [pool.submit(batching.run, None, {"images": image(v)}) for v in range(4)]
            wait_for_queue(batching, 4)
            model.gate.set()
            for future in [first, *rest]:
                future.result(5)

        assert model.batch_sizes == [1, 3, 1]

    def test_different_input_shapes_run_in_separate_batches(
        self, model: DoublingSession, batching: BatchingSession
    ) -> None:
        model.gate.clear()
        with ThreadPoolExecutor(3) as pool:
            first = pool.submit(batching.run, None, {"images": image(0)})
            model.started.wait(5)
            small = pool.submit(batching.run, None, {"images": image(1)})
            large = pool.submit(batching.run, None, {"images": image(1, size=4)})
            wait_for_queue(batching, 2)
            model.gate.set()

            assert small.result(5)[0].shape == (1, 3, 2, 2)
            assert large.result(5)[0].shape == (1, 3, 4, 4)
            first.result(5)

        assert model.batch_sizes == [1, 1, 1]

    def test_delay_gathers_requests_that_arrive_shortly_after(self, model: DoublingSession) -> None:
        session = BatchingSession(model, max_batch=4, max_delay=1.0)
        try:
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(session.run, None, {"images": image(v)}) for v in (1, 2)]
                for future in futures:
                    future.result(5)
        finally:
            session.close()

        assert model.batch_sizes == [2]

    def test_model_errors_reach_every_caller_of_the_batch(
        self, model: DoublingSession, batching: BatchingSession
    ) -> None:
        model.error = RuntimeError("bad model")

        with pytest.raises(RuntimeError, match="bad model"):
            batching.run(None, {"images": image(1)})

        model.error = None
        assert np.all(batching.run(None, {"images": image(1)})[0] == 2)

    def test_runs_after_close_are_rejected(self, model: DoublingSession) -> None:
        session = BatchingSession(model, max_batch=2)
        session.close()

        with pytest.raises(RuntimeError, match="closed"):
            session.run(None, {"images": image(1)})

    def test_detector_results_match_unbatched_inference(self) -> None:
        output = candidates((160, 160, 40, 40, 0.92, 0.01))
        plain = OnnxYoloDetector(
            "m.onnx",
            ["person", "car"],
            input_size=320,
            session_factory=lambda _: FakeSession(output),
        )

        class BatchOutput(FakeSession):
            def run(
                self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any]
            ) -> list[NDArray[np.float32]]:
                return [np.repeat(self.output, len(input_feed["images"]), axis=0)]

        session = BatchingSession(BatchOutput(output), max_batch=4)
        try:
            batched = OnnxYoloDetector(
                "m.onnx", ["person", "car"], input_size=320, session_factory=lambda _: session
            )
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            assert batched.detect(frame) == plain.detect(frame)
        finally:
            session.close()
