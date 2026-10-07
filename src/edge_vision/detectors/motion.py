from __future__ import annotations

import cv2
import numpy as np

from ..geometry import BBox
from .base import Detection, Frame

LABEL = "motion"


class MotionDetector:
    def __init__(
        self,
        *,
        min_area: float = 400.0,
        history: int = 300,
        variance_threshold: float = 32.0,
        warmup_frames: int = 10,
    ) -> None:
        if min_area <= 0:
            raise ValueError("min_area must be positive")
        if warmup_frames < 0:
            raise ValueError("warmup_frames must not be negative")
        self._min_area = min_area
        self._warmup_frames = warmup_frames
        self._seen = 0
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=variance_threshold, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: Frame) -> list[Detection]:
        mask = self._subtractor.apply(frame)
        self._seen += 1
        if self._seen <= self._warmup_frames:
            return []
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [self._to_detection(contour) for contour in contours if self._is_large(contour)]

    def _is_large(self, contour: np.ndarray) -> bool:
        return float(cv2.contourArea(contour)) >= self._min_area

    def _to_detection(self, contour: np.ndarray) -> Detection:
        x, y, w, h = cv2.boundingRect(contour)
        confidence = min(1.0, float(cv2.contourArea(contour)) / (self._min_area * 10))
        return Detection(LABEL, confidence, BBox(x, y, x + w, y + h))
