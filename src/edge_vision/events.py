from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .rules import RuleHit


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    camera_id: str
    rule_id: str
    kind: str
    track_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    occurred_at: datetime
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def _normalise_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value.astimezone(UTC)

    @classmethod
    def from_hit(cls, camera_id: str, hit: RuleHit, occurred_at: datetime) -> Event:
        box = hit.track.bbox
        return cls(
            camera_id=camera_id,
            rule_id=hit.rule_id,
            kind=hit.kind,
            track_id=hit.track.id,
            label=hit.track.label,
            confidence=round(hit.track.confidence, 4),
            bbox=(box.x1, box.y1, box.x2, box.y2),
            occurred_at=occurred_at,
            details=dict(hit.details),
        )
