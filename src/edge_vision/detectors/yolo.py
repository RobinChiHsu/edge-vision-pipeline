from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from ..geometry import BBox
from .base import Detection, Frame

PAD_VALUE = 114

logger = logging.getLogger(__name__)


class _ModelInput(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def shape(self) -> Sequence[int | str | None]: ...


class InferenceSession(Protocol):
    def get_inputs(self) -> Sequence[_ModelInput]: ...

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any]
    ) -> Sequence[NDArray[Any]]: ...


SessionFactory = Callable[[str], InferenceSession]
RuntimeSessionFactory = Callable[[str, int], InferenceSession]


@dataclass(frozen=True, slots=True)
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int
    width: int
    height: int

    def restore(self, box: BBox) -> BBox:
        return BBox(
            (box.x1 - self.pad_x) / self.scale,
            (box.y1 - self.pad_y) / self.scale,
            (box.x2 - self.pad_x) / self.scale,
            (box.y2 - self.pad_y) / self.scale,
        ).clip(self.width, self.height)


def letterbox(frame: Frame, size: int) -> tuple[NDArray[np.float32], Letterbox]:
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (size - resized_w) // 2, (size - resized_h) // 2
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = cv2.resize(
        frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
    )
    rgb = canvas[..., ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(rgb, dtype=np.float32)[np.newaxis] / 255.0
    return tensor, Letterbox(scale, pad_x, pad_y, width, height)


def decode(
    output: NDArray[np.float32], confidence: float
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.intp]]:
    predictions = output[0] if output.ndim == 3 else output
    class_scores = predictions[4:].T
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]
    keep = scores >= confidence
    cx, cy, w, h = predictions[:4, keep]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes.astype(np.float32), scores[keep], class_ids[keep]


def nms(boxes: NDArray[np.float32], scores: NDArray[np.float32], iou_threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        best, rest = order[0], order[1:]
        keep.append(int(best))
        width = np.clip(np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]), 0, None)
        height = np.clip(np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]), 0, None)
        overlap = width * height
        iou = overlap / (areas[best] + areas[rest] - overlap + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


class OnnxYoloDetector:
    def __init__(
        self,
        model_path: str,
        labels: Sequence[str],
        *,
        input_size: int = 640,
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        classes: Collection[str] | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self._session = (session_factory or SessionPool(max_batch=1).get)(model_path)
        self._input_name = self._session.get_inputs()[0].name
        self._labels = list(labels)
        self._input_size = input_size
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._classes = frozenset(classes) if classes is not None else None

    def detect(self, frame: Frame) -> list[Detection]:
        tensor, transform = letterbox(frame, self._input_size)
        output = self._session.run(None, {self._input_name: tensor})[0]
        boxes, scores, class_ids = decode(output, self._confidence)
        offsets = class_ids[:, np.newaxis].astype(np.float32) * (self._input_size + 1)
        detections = []
        for index in nms(boxes + offsets, scores, self._iou_threshold):
            label = self._label(int(class_ids[index]))
            if self._classes is None or label in self._classes:
                box = transform.restore(BBox(*boxes[index].tolist()))
                detections.append(Detection(label, float(scores[index]), box))
        return detections

    def _label(self, class_id: int) -> str:
        return self._labels[class_id] if class_id < len(self._labels) else f"class_{class_id}"


@dataclass(slots=True)
class _Request:
    output_names: tuple[str, ...] | None
    feed: Mapping[str, NDArray[Any]]
    result: Future[list[NDArray[Any]]] = field(default_factory=Future)

    def batch_key(self) -> tuple[object, ...]:
        inputs = tuple(
            (name, array.shape[1:], array.dtype.str) for name, array in self.feed.items()
        )
        return self.output_names, inputs


class BatchingSession:
    """Merges concurrent ``run`` calls from several cameras into one batched model call.

    Requests that queue up while the model is busy, or arrive within ``max_delay`` seconds of the
    first one, are stacked along the batch axis; the batch-first outputs are split back per caller.
    Requests with different input shapes are run in separate batches.
    """

    def __init__(
        self, session: InferenceSession, *, max_batch: int, max_delay: float = 0.0
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        if max_delay < 0:
            raise ValueError("max_delay must not be negative")
        self._session = session
        self._max_batch = max_batch
        self._max_delay = max_delay
        self._queue: queue.SimpleQueue[_Request | None] = queue.SimpleQueue()
        self._closed = False
        self._worker = threading.Thread(target=self._serve, name="onnx-batching", daemon=True)
        self._worker.start()

    def get_inputs(self) -> Sequence[_ModelInput]:
        return self._session.get_inputs()

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any]
    ) -> list[NDArray[Any]]:
        if self._closed:
            raise RuntimeError("session is closed")
        names = tuple(output_names) if output_names is not None else None
        request = _Request(names, {name: np.asarray(value) for name, value in input_feed.items()})
        self._queue.put(request)
        return request.result.result()

    def close(self) -> None:
        self._closed = True
        self._queue.put(None)
        self._worker.join()
        while not self._queue.empty():
            if (request := self._queue.get()) is not None:
                request.result.set_exception(RuntimeError("session is closed"))

    def _serve(self) -> None:
        while (first := self._queue.get()) is not None:
            batch, closing = self._collect(first)
            groups: dict[tuple[object, ...], list[_Request]] = {}
            for request in batch:
                groups.setdefault(request.batch_key(), []).append(request)
            for group in groups.values():
                self._run(group)
            if closing:
                return

    def _collect(self, first: _Request) -> tuple[list[_Request], bool]:
        batch = [first]
        deadline = time.monotonic() + self._max_delay
        while len(batch) < self._max_batch:
            try:
                request = self._queue.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                break
            if request is None:
                return batch, True
            batch.append(request)
        return batch, False

    def _run(self, group: list[_Request]) -> None:
        try:
            feed = {name: np.concatenate([r.feed[name] for r in group]) for name in group[0].feed}
            outputs = self._session.run(group[0].output_names, feed)
            bounds = np.cumsum([len(next(iter(r.feed.values()))) for r in group])[:-1]
            parts = [np.split(output, bounds) for output in outputs]
            for index, request in enumerate(group):
                request.result.set_result([part[index] for part in parts])
        except Exception as exc:
            for request in group:
                if not request.result.done():
                    request.result.set_exception(exc)


class SessionPool:
    """One inference session per model, shared by every camera that runs it.

    A session per camera would load the weights again and start another intra-op thread pool, so
    cameras would fight over the cores. Models exported with a dynamic batch axis additionally get
    a ``BatchingSession`` so frames from different cameras go through the model together.
    """

    def __init__(
        self,
        *,
        intra_op_threads: int | None = None,
        max_batch: int = 8,
        max_batch_delay: float = 0.0,
        session_factory: RuntimeSessionFactory | None = None,
    ) -> None:
        if intra_op_threads is not None and intra_op_threads <= 0:
            raise ValueError("intra_op_threads must be positive")
        self.intra_op_threads = intra_op_threads or max(1, (os.cpu_count() or 1) // 2)
        self._max_batch = max_batch
        self._max_batch_delay = max_batch_delay
        self._session_factory = session_factory or _onnxruntime_session
        self._sessions: dict[str, InferenceSession] = {}
        self._batching: list[BatchingSession] = []

    def get(self, model_path: str) -> InferenceSession:
        if model_path not in self._sessions:
            self._sessions[model_path] = self._open(model_path)
        return self._sessions[model_path]

    def close(self) -> None:
        for session in self._batching:
            session.close()

    def _open(self, model_path: str) -> InferenceSession:
        session = self._session_factory(model_path, self.intra_op_threads)
        if self._max_batch == 1:
            return session
        if isinstance(session.get_inputs()[0].shape[0], int):
            logger.info("%s has a fixed batch size, frames are not batched", model_path)
            return session
        batching = BatchingSession(
            session, max_batch=self._max_batch, max_delay=self._max_batch_delay
        )
        self._batching.append(batching)
        return batching


def _onnxruntime_session(model_path: str, intra_op_threads: int) -> InferenceSession:
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError(
            "OnnxYoloDetector needs onnxruntime, install edge-vision-pipeline[onnx]"
        ) from exc
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    session: InferenceSession = onnxruntime.InferenceSession(
        model_path, sess_options=options, providers=onnxruntime.get_available_providers()
    )
    return session
