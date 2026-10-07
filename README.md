# edge-vision-pipeline

[![ci](https://github.com/RobinChiHsu/edge-vision-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/RobinChiHsu/edge-vision-pipeline/actions/workflows/ci.yml)

A video analytics service that watches RTSP cameras, detects and tracks objects, applies spatial
rules such as "someone crossed the doorway" or "something stayed in the restricted area", and
publishes the resulting events through a REST API, a WebSocket stream and MQTT.

It is built on two small libraries developed alongside it:
[rtsp-supervisor](https://github.com/RobinChiHsu/rtsp-supervisor) for resilient stream ingest and
[typed-mqtt-bus](https://github.com/RobinChiHsu/typed-mqtt-bus) for typed messaging.

```
            ┌──────────────── per camera ─────────────────┐
 RTSP ──▶ rtsp-supervisor ──▶ Detector ──▶ Tracker ──▶ Rules ──▶ Event
          reconnect, latest     motion or    IoU with     zone      │
          frame, stall          YOLO (ONNX)  velocity     dwell,    │
          detection             in a thread  prediction   line      │
                                                          crossing  ▼
                                                   ┌───── fan-out ─────┐
                                                   ▼                   ▼
                                              PostgreSQL         typed-mqtt-bus
                                                   │              │         │
                                              REST API        MQTT broker  WebSocket
```

## Quick start

```bash
docker compose up --build
```

The stack contains PostgreSQL, Mosquitto, a MediaMTX server that generates two synthetic camera
streams with ffmpeg (moving shapes, so no model download or video file is needed) and the service.

```bash
curl localhost:8000/cameras                         # stream health, latency, counters per rule
curl "localhost:8000/events?camera_id=entrance"     # stored events, newest first
websocat ws://localhost:8000/events/live            # live events
mosquitto_sub -t 'edge-vision/cameras/+/events'     # the same events over MQTT
```

## What happens to a frame

1. **Ingest.** `rtsp-supervisor` reads the stream in a worker thread, reconnects with backoff when
   the camera disappears and always hands over the newest frame. When analysis is slower than the
   camera, stale frames are skipped instead of queued, so events never lag behind reality.
2. **Detection.** Two detectors implement the same protocol:
   - `MotionDetector`: background subtraction (MOG2) with morphological clean-up. Cheap, needs no
     model, used by the demo.
   - `OnnxYoloDetector`: runs a YOLOv8/YOLO11 ONNX export. Letterboxing, output decoding and
     class-aware non-maximum suppression are plain NumPy functions with their own tests.

   Detection runs in a thread so the event loop keeps serving the API. `max_fps` caps the work per
   camera.
3. **Tracking.** `IouTracker` matches detections to tracks by IoU against a constant-velocity
   prediction, which keeps identities stable for fast objects at low frame rates. Tracks need
   `min_hits` confirmations before they are reported and survive `max_missed` frames of occlusion.
   The tracker returns immutable snapshots, so rules cannot corrupt its state.
4. **Rules.**
   - `ZoneRule` reports an object inside a polygon once per visit, optionally only after it has
     stayed for `min_dwell` seconds, and exposes the current occupancy.
   - `LineRule` counts crossings of a segment by direction. A per-object cooldown suppresses
     double counting when an object jitters on the line.

   Objects are located by the bottom centre of their box, which approximates where they touch the
   ground.
5. **Delivery.** Each event is handed to a fan-out with one bounded queue and worker per sink.
   A slow or failing sink (say, the database during a failover) retries with backoff and never
   blocks the other sinks or the video pipeline. On shutdown queues are drained before exit.

### Line direction

A line goes from `start` to `end`. Imagine standing on `start` and looking at `end`: an object
that moves onto your left-hand side crossed to the `left`. In the demo the doorway runs from
`(320, 0)` down to `(320, 360)`, so an object walking left to right on screen crosses to the
`left`. Set `direction: left`, `right` or `both`.

## Configuration

```yaml
database_url: postgresql+asyncpg://edge:${POSTGRES_PASSWORD}@postgres:5432/edge_vision

mqtt:                       # optional; without it events stay in-process
  host: mosquitto

detector:                   # default for all cameras
  type: motion
  min_area: 800

tracker:
  min_hits: 2
  max_missed: 10

cameras:
  - id: entrance
    url: rtsp://viewer:${CAMERA_PASSWORD}@192.168.1.20/stream1
    max_fps: 10
    rules:
      - type: line
        id: doorway
        start: [320, 0]
        end: [320, 360]
      - type: zone
        id: restricted
        polygon: [[480, 0], [640, 0], [640, 360], [480, 360]]
        labels: [person]
        min_dwell: 2

  - id: parking
    url: rtsp://192.168.1.21/stream1
    detector:                # per-camera override
      type: yolo
      model: models/yolo11n.onnx
      labels: [person, bicycle, car]
      classes: [car]
```

`${NAME}` and `${NAME:-default}` are expanded from the environment, so credentials do not have to
live in the file. Camera URLs and the database URL are stored as secrets and never appear in logs,
`repr` or API responses. The configuration is validated completely at start-up: unknown keys,
duplicate ids, degenerate polygons and zero-length lines are rejected with a clear message.

## API

| Endpoint | Description |
| --- | --- |
| `GET /healthz` | Liveness |
| `GET /readyz` | Database reachability and camera health, `503` when the database is down |
| `GET /cameras`, `GET /cameras/{id}` | Stream state, reconnects, last error, frames processed and skipped, inference latency, zone occupancy, line counts |
| `GET /events` | Filter by `camera_id`, `rule_id`, `kind`, `label`, `since`, `until`; keyset pagination with `limit` and `cursor` |
| `GET /events/{id}` | A single event |
| `GET /delivery` | Delivered, failed and dropped events per sink |
| `WS /events/live?camera_id=` | Live events, optionally for one camera |

Pagination is keyset based on `(occurred_at, id)`, so deep pages cost the same as the first one and
results stay consistent while new events arrive.

Live events go through the same bus as MQTT. With a broker configured, every instance of the
service sees the events of all instances; without one, an in-memory broker keeps a single-node
deployment free of extra infrastructure.

## Testing

```bash
uv sync
uv run pytest
EDGE_VISION_TEST_POSTGRES_URL=postgresql+asyncpg://user:pass@localhost/test uv run pytest
```

- **Unit tests** cover geometry, detectors, tracker, rules, storage, delivery, configuration and API.
- **Property-based tests** (Hypothesis) check invariants such as IoU symmetry, NMS never keeping
  overlapping boxes and line crossings reversing direction when the movement is reversed. They
  found two real bugs in line crossing: an asymmetry caused by floating point rounding, and
  crossings that were missed when an object's anchor landed exactly on the line.
- **Integration tests** run the storage layer against both SQLite and PostgreSQL, and drive a
  synthetic moving object through rtsp-supervisor, the motion detector, the tracker and a line rule.
- **End-to-end** in CI starts the full docker compose stack and waits for events from both synthetic
  cameras over HTTP and MQTT.

Static checks: `mypy --strict` and `ruff`.

## Running without Docker

```bash
uv sync --extra onnx            # the extra is only needed for YOLO
EDGE_VISION_CAMERA_URL=rtsp://... uv run edge-vision --config config/local.yaml
```

`config/local.yaml` uses SQLite and the in-memory bus.

## Limitations

- One process handles all configured cameras. CPU-bound detection runs in threads, which is fine
  for motion detection and ONNX Runtime (it releases the GIL) but would need worker processes for
  pure-Python detectors.
- Rules evaluate in image coordinates; there is no camera calibration or ground-plane mapping.
- Schema creation uses `create_all`. A real deployment would add migrations.

## License

MIT
