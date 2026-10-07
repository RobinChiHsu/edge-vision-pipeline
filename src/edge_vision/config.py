from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from .detectors import Detector
from .detectors.motion import MotionDetector
from .geometry import Line, Point, Polygon
from .rules import Direction, LineRule, Rule, ZoneRule
from .tracking import IouTracker

_ENV_REFERENCE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")

Coordinate = tuple[float, float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MotionDetectorConfig(_Strict):
    type: Literal["motion"] = "motion"
    min_area: float = Field(default=400.0, gt=0)
    warmup_frames: int = Field(default=10, ge=0)

    def build(self) -> Detector:
        return MotionDetector(min_area=self.min_area, warmup_frames=self.warmup_frames)


class YoloDetectorConfig(_Strict):
    type: Literal["yolo"]
    model: Path
    labels: list[str]
    input_size: int = Field(default=640, gt=0)
    confidence: float = Field(default=0.35, gt=0, le=1)
    iou_threshold: float = Field(default=0.45, gt=0, le=1)
    classes: list[str] | None = None

    def build(self) -> Detector:
        from .detectors.yolo import OnnxYoloDetector

        return OnnxYoloDetector(
            str(self.model),
            self.labels,
            input_size=self.input_size,
            confidence=self.confidence,
            iou_threshold=self.iou_threshold,
            classes=self.classes,
        )


DetectorConfig = Annotated[MotionDetectorConfig | YoloDetectorConfig, Field(discriminator="type")]


class TrackerConfig(_Strict):
    iou_threshold: float = Field(default=0.3, gt=0, le=1)
    max_missed: int = Field(default=15, ge=0)
    min_hits: int = Field(default=3, ge=1)

    def build(self) -> IouTracker:
        return IouTracker(
            iou_threshold=self.iou_threshold, max_missed=self.max_missed, min_hits=self.min_hits
        )


class ZoneRuleConfig(_Strict):
    type: Literal["zone"]
    id: str
    polygon: list[Coordinate] = Field(min_length=3)
    labels: list[str] | None = None
    min_dwell: float = Field(default=0.0, ge=0)

    def build(self) -> Rule:
        return ZoneRule(
            self.id, Polygon.of(self.polygon), labels=self.labels, min_dwell=self.min_dwell
        )


class LineRuleConfig(_Strict):
    type: Literal["line"]
    id: str
    start: Coordinate
    end: Coordinate
    direction: Direction = Direction.BOTH
    labels: list[str] | None = None
    cooldown: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def _non_zero_length(self) -> Self:
        if self.start == self.end:
            raise ValueError(f"line {self.id} has zero length")
        return self

    def build(self) -> Rule:
        return LineRule(
            self.id,
            Line(Point(*self.start), Point(*self.end)),
            direction=self.direction,
            labels=self.labels,
            cooldown=self.cooldown,
        )


RuleConfig = Annotated[ZoneRuleConfig | LineRuleConfig, Field(discriminator="type")]


class CameraConfig(_Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    url: SecretStr
    max_fps: float | None = Field(default=5.0, gt=0)
    detector: DetectorConfig | None = None
    rules: list[RuleConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_rule_ids(self) -> Self:
        _require_unique([rule.id for rule in self.rules], f"rule id on camera {self.id}")
        return self


class MqttConfig(_Strict):
    host: str
    port: int = Field(default=1883, gt=0, le=65535)
    username: str | None = None
    password: SecretStr | None = None


class AppConfig(_Strict):
    database_url: SecretStr
    mqtt: MqttConfig | None = None
    detector: DetectorConfig = Field(default_factory=MotionDetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    cameras: list[CameraConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_camera_ids(self) -> Self:
        _require_unique([camera.id for camera in self.cameras], "camera id")
        return self

    def detector_for(self, camera: CameraConfig) -> DetectorConfig:
        return camera.detector or self.detector


class MissingEnvironmentVariable(ValueError):
    pass


def expand_environment(text: str, environ: dict[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ

    def substitute(match: re.Match[str]) -> str:
        name, default = match.group("name"), match.group("default")
        if name in values:
            return values[name]
        if default is not None:
            return default
        raise MissingEnvironmentVariable(f"environment variable {name} is not set")

    return _ENV_REFERENCE.sub(substitute, text)


def load_config(path: Path, environ: dict[str, str] | None = None) -> AppConfig:
    raw = yaml.safe_load(expand_environment(path.read_text(), environ))
    return AppConfig.model_validate(raw)


def _require_unique(values: list[str], what: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {what}: {', '.join(duplicates)}")
