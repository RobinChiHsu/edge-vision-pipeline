from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from ..geometry import BBox
from .base import Detection, Frame

PAD_VALUE = 114


class _ModelInput(Protocol):
    @property
    def name(self) -> str: ...


class InferenceSession(Protocol):
    def get_inputs(self) -> Sequence[_ModelInput]: ...

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any]
    ) -> Sequence[NDArray[Any]]: ...


SessionFactory = Callable[[str], InferenceSession]


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
        self._session = (session_factory or _onnxruntime_session)(model_path)
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


def _onnxruntime_session(model_path: str) -> InferenceSession:
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError(
            "OnnxYoloDetector needs onnxruntime, install edge-vision-pipeline[onnx]"
        ) from exc
    session: InferenceSession = onnxruntime.InferenceSession(
        model_path, providers=onnxruntime.get_available_providers()
    )
    return session
