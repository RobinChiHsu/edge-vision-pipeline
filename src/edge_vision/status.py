from __future__ import annotations

from pydantic import BaseModel


class RuleStatus(BaseModel):
    id: str
    type: str
    occupancy: int | None = None
    counts: dict[str, int] | None = None


class CameraStatus(BaseModel):
    id: str
    stream_state: str
    healthy: bool
    pipeline_running: bool
    frames_read: int
    frames_dropped: int
    reconnect_attempts: int
    last_error: str | None
    frames_processed: int
    frames_skipped: int
    detector_errors: int
    events_emitted: int
    active_tracks: int
    inference_ms: float | None
    rules: list[RuleStatus]
