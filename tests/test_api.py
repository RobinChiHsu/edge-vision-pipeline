from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx2 import ASGITransport, AsyncClient
from starlette.websockets import WebSocketDisconnect
from typed_mqtt_bus import Bus, InMemoryBroker

from edge_vision.api import Services, create_app
from edge_vision.sinks import BusSink, DeliveryStats
from edge_vision.status import CameraStatus, RuleStatus
from edge_vision.storage import SqlEventRepository
from tests.helpers import make_event


def camera_status(camera_id: str = "gate", healthy: bool = True) -> CameraStatus:
    return CameraStatus(
        id=camera_id,
        stream_state="streaming",
        healthy=healthy,
        pipeline_running=True,
        frames_read=10,
        frames_dropped=1,
        reconnect_attempts=0,
        last_error=None,
        frames_processed=9,
        frames_skipped=0,
        detector_errors=0,
        events_emitted=2,
        active_tracks=1,
        inference_ms=4.2,
        rules=[RuleStatus(id="door", type="line", counts={"left": 2, "right": 0})],
    )


class FakeServices:
    def __init__(self, repository: SqlEventRepository, bus: Bus) -> None:
        self.repository = repository
        self.bus = bus
        self.statuses = [camera_status("gate"), camera_status("dock", healthy=False)]

    def camera_statuses(self) -> list[CameraStatus]:
        return self.statuses

    def delivery_stats(self) -> dict[str, DeliveryStats]:
        return {"database": DeliveryStats(delivered=3)}


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    @asynccontextmanager
    async def services() -> AsyncIterator[Services]:
        repository = SqlEventRepository.from_url(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
        await repository.create_schema()
        await repository.add_many(
            [
                make_event(0, track_id=1),
                make_event(1, track_id=2, camera_id="dock", kind="zone_entered"),
                make_event(2, track_id=3),
            ]
        )
        async with Bus(InMemoryBroker().client()) as bus:
            yield FakeServices(repository, bus)
        await repository.close()

    return create_app(services)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/healthz")

    assert response.json() == {"status": "ok"}


async def test_ready_reports_database_and_cameras(client: AsyncClient) -> None:
    response = await client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"database": "ok", "cameras": {"gate": True, "dock": False}}


async def test_not_ready_when_database_is_down(client: AsyncClient, app: FastAPI) -> None:
    await app.state.services.repository.close()

    async def broken() -> None:
        raise ConnectionError("db down")

    app.state.services.repository.ping = broken

    response = await client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["database"] == "unavailable"


async def test_lists_cameras_and_finds_one(client: AsyncClient) -> None:
    listed = (await client.get("/cameras")).json()
    single = await client.get("/cameras/gate")
    missing = await client.get("/cameras/nope")

    assert [camera["id"] for camera in listed] == ["gate", "dock"]
    assert single.json()["rules"][0]["counts"] == {"left": 2, "right": 0}
    assert missing.status_code == 404


async def test_delivery_stats(client: AsyncClient) -> None:
    response = await client.get("/delivery")

    assert response.json() == {"database": {"delivered": 3, "failed": 0, "dropped": 0}}


async def test_lists_events_with_filters_and_pagination(client: AsyncClient) -> None:
    first = (await client.get("/events", params={"camera_id": "gate", "limit": 1})).json()
    second = (
        await client.get("/events", params={"camera_id": "gate", "cursor": first["next_cursor"]})
    ).json()

    assert [item["track_id"] for item in first["items"]] == [3]
    assert [item["track_id"] for item in second["items"]] == [1]
    assert second["next_cursor"] is None


async def test_time_window_filter(client: AsyncClient) -> None:
    response = await client.get(
        "/events", params={"since": "2026-03-01T08:00:01Z", "until": "2026-03-01T08:00:02+00:00"}
    )

    assert [item["track_id"] for item in response.json()["items"]] == [2]


@pytest.mark.parametrize(
    ("params", "status_code"),
    [
        ({"cursor": "garbage!"}, 400),
        ({"limit": 0}, 422),
        ({"limit": 501}, 422),
        ({"since": "2026-03-01T08:00:00"}, 422),
    ],
)
async def test_rejects_bad_queries(
    client: AsyncClient, params: dict[str, str | int], status_code: int
) -> None:
    response = await client.get("/events", params=params)

    assert response.status_code == status_code


async def test_gets_single_event(client: AsyncClient) -> None:
    listed = (await client.get("/events", params={"limit": 1})).json()["items"][0]

    found = await client.get(f"/events/{listed['id']}")
    missing = await client.get("/events/00000000-0000-0000-0000-000000000000")
    invalid = await client.get("/events/not-a-uuid")

    assert found.json() == listed
    assert missing.status_code == 404
    assert invalid.status_code == 422


def test_live_events_stream_only_requested_camera(app: FastAPI) -> None:
    with TestClient(app) as client, client.websocket_connect("/events/live?camera_id=gate") as ws:
        sink = BusSink(app.state.services.bus)
        assert client.portal is not None
        client.portal.call(sink.emit, make_event(5, camera_id="dock", track_id=50))
        client.portal.call(sink.emit, make_event(6, camera_id="gate", track_id=60))

        received = ws.receive_json()

    assert (received["camera_id"], received["track_id"]) == ("gate", 60)


def test_live_events_without_filter_and_clean_disconnect(app: FastAPI) -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/events/live") as ws:
            assert client.portal is not None
            client.portal.call(BusSink(app.state.services.bus).emit, make_event(7, track_id=70))
            assert ws.receive_json()["track_id"] == 70
        bus = app.state.services.bus
        assert client.portal.call(lambda: _subscriber_count(bus)) == 0


async def _subscriber_count(bus: Bus) -> int:
    return len(bus._subscriptions)


def test_live_events_rejects_invalid_camera_id(app: FastAPI) -> None:
    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect) as closed,
        client.websocket_connect("/events/live?camera_id=a/b") as ws,
    ):
        ws.receive_json()

    assert closed.value.code == 1008


def test_live_events_close_when_bus_stops(app: FastAPI) -> None:
    with TestClient(app) as client, client.websocket_connect("/events/live") as ws:
        assert client.portal is not None
        bus = app.state.services.bus
        client.portal.call(bus.__aexit__, None, None, None)

        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()

    assert closed.value.code == 1001
