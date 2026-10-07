from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime
from typing import Annotated, Any, Protocol
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, status
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect
from typed_mqtt_bus import Bus

from .events import Event
from .sinks import EVENTS_TOPIC, DeliveryStats
from .status import CameraStatus
from .storage import MAX_PAGE_SIZE, EventPage, EventQuery, EventRepository, InvalidCursor


class Services(Protocol):
    @property
    def repository(self) -> EventRepository: ...

    @property
    def bus(self) -> Bus: ...

    def camera_statuses(self) -> list[CameraStatus]: ...

    def delivery_stats(self) -> dict[str, DeliveryStats]: ...


ServicesFactory = Callable[[], AbstractAsyncContextManager[Services]]


def create_app(services_factory: ServicesFactory) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with services_factory() as services:
            app.state.services = services
            yield

    app = FastAPI(title="edge-vision-pipeline", version="0.1.0", lifespan=lifespan)
    _register_routes(app)
    return app


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDependency = Annotated[Services, Depends(get_services)]


def _register_routes(app: FastAPI) -> None:
    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready(services: ServicesDependency) -> JSONResponse:
        cameras = {camera.id: camera.healthy for camera in services.camera_statuses()}
        try:
            await services.repository.ping()
        except Exception:
            body: dict[str, Any] = {"database": "unavailable", "cameras": cameras}
            return JSONResponse(body, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return JSONResponse({"database": "ok", "cameras": cameras})

    @app.get("/cameras")
    async def cameras(services: ServicesDependency) -> list[CameraStatus]:
        return services.camera_statuses()

    @app.get("/cameras/{camera_id}")
    async def camera(camera_id: str, services: ServicesDependency) -> CameraStatus:
        for item in services.camera_statuses():
            if item.id == camera_id:
                return item
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"camera {camera_id} not found")

    @app.get("/delivery")
    async def delivery(services: ServicesDependency) -> dict[str, DeliveryStats]:
        return services.delivery_stats()

    @app.get("/events")
    async def list_events(
        services: ServicesDependency,
        camera_id: str | None = None,
        rule_id: str | None = None,
        kind: str | None = None,
        label: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
        cursor: str | None = None,
    ) -> EventPage:
        for name, value in (("since", since), ("until", until)):
            if value is not None and value.tzinfo is None:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, f"{name} needs a timezone"
                )
        query = EventQuery(
            camera_id=camera_id,
            rule_id=rule_id,
            kind=kind,
            label=label,
            since=since,
            until=until,
            limit=limit,
            cursor=cursor,
        )
        try:
            return await services.repository.list(query)
        except InvalidCursor as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    @app.get("/events/{event_id}")
    async def get_event(event_id: UUID, services: ServicesDependency) -> Event:
        event = await services.repository.get(event_id)
        if event is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"event {event_id} not found")
        return event

    @app.websocket("/events/live")
    async def live_events(websocket: WebSocket, camera_id: str | None = None) -> None:
        services: Services = websocket.app.state.services
        try:
            subscription = services.bus.subscribe(
                EVENTS_TOPIC, params={"camera_id": camera_id} if camera_id else None
            )
        except ValueError:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        async with subscription:
            await websocket.accept()
            await _forward(websocket, subscription)


async def _forward(websocket: WebSocket, events: AsyncIterator[Any]) -> None:
    disconnected = asyncio.create_task(_wait_for_disconnect(websocket))
    try:
        while True:
            next_event = asyncio.ensure_future(anext(events))
            done, _ = await asyncio.wait(
                {next_event, disconnected}, return_when=asyncio.FIRST_COMPLETED
            )
            if disconnected in done:
                next_event.cancel()
                return
            try:
                envelope = next_event.result()
            except StopAsyncIteration:
                await websocket.close(code=status.WS_1001_GOING_AWAY)
                return
            await websocket.send_text(envelope.message.model_dump_json())
    finally:
        disconnected.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await disconnected


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    with contextlib.suppress(WebSocketDisconnect):
        while True:
            await websocket.receive_text()
