from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.asyncio import create_async_engine

from edge_vision.events import Event
from edge_vision.storage import (
    EventQuery,
    InvalidCursor,
    SqlEventRepository,
    UtcDateTime,
    metadata,
)

POSTGRES_URL = os.environ.get("EDGE_VISION_TEST_POSTGRES_URL", "")
BASE = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def event(offset: int = 0, **overrides: object) -> Event:
    fields: dict[str, object] = {
        "camera_id": "gate",
        "rule_id": "entrance",
        "kind": "line_crossed",
        "track_id": 1,
        "label": "person",
        "confidence": 0.91,
        "bbox": (1.5, 2.0, 30.0, 80.25),
        "occurred_at": BASE + timedelta(seconds=offset),
        "details": {"direction": "left", "count": 3},
    }
    return Event.model_validate({**fields, **overrides})


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param(
            "postgres",
            marks=pytest.mark.skipif(
                not POSTGRES_URL, reason="set EDGE_VISION_TEST_POSTGRES_URL to run"
            ),
        ),
    ]
)
async def repository(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[SqlEventRepository]:
    url = (
        f"sqlite+aiosqlite:///{tmp_path / 'events.db'}"
        if request.param == "sqlite"
        else POSTGRES_URL
    )
    engine = create_async_engine(url)
    async with engine.begin() as connection:
        await connection.run_sync(metadata.drop_all)
    repository = SqlEventRepository(engine)
    await repository.create_schema()
    yield repository
    await repository.close()


async def test_round_trips_every_field(repository: SqlEventRepository) -> None:
    original = event()
    await repository.add(original)

    assert await repository.get(original.id) == original


async def test_missing_event_is_none(repository: SqlEventRepository) -> None:
    assert await repository.get(uuid4()) is None


async def test_lists_newest_first(repository: SqlEventRepository) -> None:
    await repository.add_many([event(0), event(20), event(10)])

    page = await repository.list(EventQuery())

    assert [item.occurred_at.second for item in page.items] == [20, 10, 0]
    assert page.next_cursor is None


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ({"camera_id": "dock"}, [3]),
        ({"rule_id": "loading-bay"}, [3]),
        ({"kind": "zone_entered"}, [3, 2]),
        ({"label": "car"}, [2]),
        ({"since": BASE + timedelta(seconds=2)}, [3, 2]),
        ({"until": BASE + timedelta(seconds=2)}, [1]),
    ],
)
async def test_filters(
    repository: SqlEventRepository, query: dict[str, object], expected: list[int]
) -> None:
    await repository.add_many(
        [
            event(1, track_id=1),
            event(2, track_id=2, kind="zone_entered", label="car"),
            event(3, track_id=3, kind="zone_entered", camera_id="dock", rule_id="loading-bay"),
        ]
    )

    page = await repository.list(EventQuery.model_validate(query))

    assert [item.track_id for item in page.items] == expected


async def test_paginates_without_gaps_or_duplicates(repository: SqlEventRepository) -> None:
    stored = [event(i // 3, track_id=i) for i in range(11)]
    await repository.add_many(stored)
    seen: list[Event] = []
    cursor: str | None = None

    for _ in range(10):
        page = await repository.list(EventQuery(limit=4, cursor=cursor))
        seen.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            break

    assert len(seen) == 11
    assert {item.id for item in seen} == {item.id for item in stored}
    keys = [(item.occurred_at, item.id) for item in seen]
    assert keys == sorted(keys, reverse=True)


async def test_timestamps_come_back_in_utc(repository: SqlEventRepository) -> None:
    taipei = timezone(timedelta(hours=8))
    original = event(occurred_at=datetime(2026, 1, 1, 20, 0, tzinfo=taipei))
    await repository.add(original)

    stored = await repository.get(original.id)

    assert stored is not None
    assert stored.occurred_at == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert stored.occurred_at.tzinfo == UTC


@pytest.mark.parametrize(
    "cursor", ["not-base64!", "bm90IGpzb24", "WyIyMDI2LTAxLTAxVDAwOjAwOjAwIiwgIngiXQ"]
)
async def test_rejects_malformed_cursor(repository: SqlEventRepository, cursor: str) -> None:
    with pytest.raises(InvalidCursor):
        await repository.list(EventQuery(cursor=cursor))


async def test_ping_and_empty_batch(repository: SqlEventRepository) -> None:
    await repository.ping()
    await repository.add_many([])

    assert (await repository.list(EventQuery())).items == []


def test_event_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        event(occurred_at=datetime(2026, 1, 1))


def test_from_url_builds_repository(tmp_path: Path) -> None:
    repository = SqlEventRepository.from_url(f"sqlite+aiosqlite:///{tmp_path / 'x.db'}")

    assert isinstance(repository, SqlEventRepository)


def test_utc_column_passes_null_through() -> None:
    column = UtcDateTime()
    dialect = sqlite.dialect()

    assert column.process_bind_param(None, dialect) is None
    assert column.process_result_value(None, dialect) is None
