from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Dialect,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    TypeDecorator,
    Uuid,
    and_,
    insert,
    or_,
    select,
    text,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .events import Event

MAX_PAGE_SIZE = 500


class UtcDateTime(TypeDecorator[datetime]):
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        value = value.astimezone(UTC)
        return value.replace(tzinfo=None) if dialect.name == "sqlite" else value

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


metadata = MetaData()

events = Table(
    "events",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("camera_id", String(64), nullable=False),
    Column("rule_id", String(64), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("track_id", Integer, nullable=False),
    Column("label", String(64), nullable=False),
    Column("confidence", Float, nullable=False),
    Column("bbox", JSON, nullable=False),
    Column("occurred_at", UtcDateTime, nullable=False),
    Column("details", JSON, nullable=False),
    Index("ix_events_occurred_at_id", "occurred_at", "id"),
    Index("ix_events_camera_occurred_at", "camera_id", "occurred_at"),
)


class EventQuery(BaseModel):
    camera_id: str | None = None
    rule_id: str | None = None
    kind: str | None = None
    label: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = Field(default=50, ge=1, le=MAX_PAGE_SIZE)
    cursor: str | None = None


class EventPage(BaseModel):
    items: list[Event]
    next_cursor: str | None


class InvalidCursor(ValueError):
    pass


class EventRepository(Protocol):
    async def add(self, event: Event) -> None: ...

    async def get(self, event_id: UUID) -> Event | None: ...

    async def list(self, query: EventQuery) -> EventPage: ...

    async def ping(self) -> None: ...


class SqlEventRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @classmethod
    def from_url(cls, url: str) -> SqlEventRepository:
        return cls(create_async_engine(url, pool_pre_ping=True))

    async def create_schema(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    async def ping(self) -> None:
        async with self._engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def add(self, event: Event) -> None:
        await self.add_many([event])

    async def add_many(self, items: Iterable[Event]) -> None:
        rows = [_to_row(event) for event in items]
        if rows:
            async with self._engine.begin() as connection:
                await connection.execute(insert(events), rows)

    async def get(self, event_id: UUID) -> Event | None:
        async with self._engine.connect() as connection:
            result = await connection.execute(select(events).where(events.c.id == event_id))
            row = result.mappings().first()
        return _from_row(row) if row is not None else None

    async def list(self, query: EventQuery) -> EventPage:
        statement = (
            select(events)
            .where(*_filters(query))
            .order_by(events.c.occurred_at.desc(), events.c.id.desc())
            .limit(query.limit + 1)
        )
        async with self._engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
        items = [_from_row(row) for row in rows[: query.limit]]
        has_more = len(rows) > query.limit
        return EventPage(items=items, next_cursor=_encode_cursor(items[-1]) if has_more else None)


def _filters(query: EventQuery) -> list[Any]:
    conditions: list[Any] = [
        column == value
        for column, value in (
            (events.c.camera_id, query.camera_id),
            (events.c.rule_id, query.rule_id),
            (events.c.kind, query.kind),
            (events.c.label, query.label),
        )
        if value is not None
    ]
    if query.since is not None:
        conditions.append(events.c.occurred_at >= query.since)
    if query.until is not None:
        conditions.append(events.c.occurred_at < query.until)
    if query.cursor is not None:
        occurred_at, event_id = _decode_cursor(query.cursor)
        conditions.append(
            or_(
                events.c.occurred_at < occurred_at,
                and_(events.c.occurred_at == occurred_at, events.c.id < event_id),
            )
        )
    return conditions


def _encode_cursor(event: Event) -> str:
    payload = json.dumps([event.occurred_at.isoformat(), str(event.id)])
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        occurred_at, event_id = json.loads(base64.urlsafe_b64decode(padded))
        parsed = datetime.fromisoformat(occurred_at)
        if parsed.tzinfo is None:
            raise ValueError("cursor timestamp is not timezone-aware")
        return parsed, UUID(event_id)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise InvalidCursor("malformed cursor") from exc


def _to_row(event: Event) -> dict[str, Any]:
    return {**event.model_dump(), "bbox": list(event.bbox)}


def _from_row(row: Any) -> Event:
    return Event.model_validate(dict(row))
