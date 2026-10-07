from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..geometry import BBox

Frame: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    bbox: BBox


class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...
