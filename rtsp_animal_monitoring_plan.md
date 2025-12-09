# RTSP Animal Monitoring Implementation Plan

## 1. Goals & Success Criteria
- Detect animals in near real-time from an RTSP stream using the latest YOLO model on a Jetson Nano.
- Identify animal class, capture short clips around each detection, and push notifications via Pushover.
- Keep the pipeline resilient to RTSP dropouts and provide basic observability (metrics/logs).
- Support multiple RTSP feeds concurrently with per-camera alerting, clip retention, and resource controls.
- Ensure ONVIF-compliant discovery/control so cameras can be auto-registered and verified remotely.

## 2. Assumptions & Constraints
- Hardware: Jetson Nano 4GB with CSI/USB camera input converted to RTSP, running JetPack 5.x.
- OS: Ubuntu 20.04 LTS (Jetson default) with NVIDIA drivers, CUDA, TensorRT, and DeepStream SDK available.
- Stream characteristics: H.264/H.265 up to 1080p @ 30 FPS; latency tolerance ≤ 2 s.
- Network: Jetson has outbound HTTPS access for Pushover and optional clip upload (S3/NAS).
- Cameras: Target 2–4 RTSP feeds per Jetson Nano (tunable); scale horizontally with additional devices or remote inference nodes as needed.
- Cameras expose ONVIF Profile S/T endpoints for discovery, status, and optional PTZ control on the same network segment.
- Storage: Dedicated 1 TB SSD mounted on the Nano (e.g., `/mnt/wildlife_ssd`) for clips/logs, with SD/eMMC reserved for OS.

## 3. High-Level Architecture
1. **RTSP Ingest**: GStreamer (or FFmpeg) pipeline running continuously, decoding frames on GPU.
2. **Inference Engine**: YOLO model (e.g., YOLOv8) converted to TensorRT via Ultralytics export or DeepStream nvinfer plugin.
3. **Event Logic**: Detection aggregator decides when an animal event starts/ends and tags species/confidence.
4. **Clip Buffer**: Circular frame buffer retains pre/post event footage, encodes to MP4/H.264 when an event fires.
5. **Notification Service**: Sends Pushover message with metadata and clip link; optional upload to S3/NAS.
6. **Persistence & Monitoring**: SQLite/JSON logs, Prometheus-style metrics endpoint, systemd watchdogs.
7. **Multi-Camera Orchestrator**: Manages per-stream workers, schedules GPU time, and aggregates alerts for centralized control.

```
RTSP Cameras (A…N) → Stream Workers → YOLO Inference → Event Logic
                           ↓                ↓
                  Circular Buffer     Notification Hub
                           ↓                ↓
                SSD Storage / Logs ← Persistence Layer
```

## 4. Detailed Components
### 4.1 RTSP Capture
- Use GStreamer: `rtspsrc latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! appsink`.
- Configure reconnect logic (timeout handling, exponential backoff, watchdog restart).
- Optionally leverage NVIDIA DeepStream `deepstream-app` pipeline if more convenient.
- Pair each RTSP URL with ONVIF-derived connection info to auto-refresh credentials and verify stream health/metadata before launching workers.

### 4.2 YOLO Inference
- Train/fine-tune YOLO model if needed for specific animal classes; export to ONNX then TensorRT.
- Run via:
  - **DeepStream**: `nvinfer` plugin referencing TensorRT engine, enabling batching and GPU inference.
  - **Ultralytics + TensorRT**: Python loop loading TensorRT engine through `tensorrt` + `pycuda` for flexibility.
- Maintain minimum 15 FPS throughput; adjust input resolution or use INT8 quantization if needed.

### 4.3 Detection Event Logic
- For each frame, collect detections (class, bbox, confidence).
- Apply confidence threshold (default 0.5) and class whitelist for animals of interest.
- Apply confidence threshold (default 0.5) and class whitelist for animals of interest; honor an exclusion list that suppresses detections for nuisance species (e.g., squirrels) before notifications are generated.
- Use temporal smoothing (e.g., require ≥3 positive frames within 1 s) to avoid false positives.
- Define event lifecycle:
  1. **Start**: first qualifying detection triggers clip buffering and metadata capture.
  2. **Active**: keep recording while detections continue; extend timeout (e.g., 3 s) after last detection.
  3. **End**: finalize metadata (species label, max confidence, duration, bounding boxes snapshot).

### 4.4 Clip Buffering & Encoding
- Keep a circular buffer (e.g., `collections.deque`) with raw frames for the previous N seconds (default 5 s).
- On event start, duplicate buffered frames + ongoing frames until event end.
- Encode via `ffmpeg` or GStreamer `qtmux ! filesink` to MP4 (H.264) with timestamp-based filenames.
- Store clips on the SSD under `/mnt/wildlife_ssd/clips/YYYY/MM/DD/animal_class/`.
- Enforce retention policy (e.g., prune older than 30 days or when disk >80%).

### 4.5 Storage Management & Cleanup
- Mount 1 TB SSD with ext4/XFS and enable `noatime` to reduce wear; verify UUID in `/etc/fstab` for boot persistence.
- Allocate top-level directories for `clips/`, `logs/`, `models/`; expose usage via `du` metrics for observability.
- Implement cleanup daemon (Python or bash) that runs hourly via systemd timer:
   - Target steady-state utilization ≤80% (≈800 GB) while preserving minimum 7 days of clips per species.
   - Evict oldest clips first using a two-pass scan: delete files past retention window, then trim by disk quota.
   - Purge orphaned temp files and compress/archive logs older than 30 days.
- Integrate `systemd` service with `RequiresMountsFor=/mnt/wildlife_ssd` to ensure detector starts only when SSD is available.
- Emit alerts (Pushover + log) if available space <15% so field team can investigate before capture stops.

### 4.6 Notification Pipeline
- After clip file is ready:
  - Upload to object storage (S3/NAS) or expose via local web server for remote access.
  - Compose Pushover payload: title, body (species, confidence, timestamp), priority (e.g., high for rare species).
  - Include URL or attachment of clip (Pushover supports up to 2.5 MB); otherwise send link.
- Add rate limiting to avoid notification storms (e.g., one per species per minute) and skip notifications for species present in the exclusion list while still logging the event for auditing, if desired.

### 4.7 Persistence & Observability
- **Metadata log**: SQLite table `events` with columns `(id, species, confidence, start_ts, end_ts, clip_path, bbox_json)`.
- **Metrics**: expose counts and latency via simple HTTP server (`prometheus_client`) for remote scraping.
- **System health**: systemd unit with `Restart=on-failure`, optional hardware watchdog, log rotation via `logrotate`.

### 4.8 Multi-Camera Coordination
- Declare cameras in a version-controlled YAML/JSON config (`/etc/wildlife/cameras.yml`) specifying RTSP URL, location, target species, per-camera exclusion list, and notification routing.
- Spin up one stream worker per camera (Docker container or systemd template unit `detector@cam.service`) so faults are isolated.
- Use a central scheduler thread to monitor aggregate GPU load (TensorRT contexts) and throttle per-stream inference rate when utilization exceeds threshold (e.g., 85%).
- Maintain per-camera buffers and detection state machines while sharing reusable components (model weights, encoding pools) to minimize memory overhead.
- Aggregate detections into a unified event bus (Redis streams, ZeroMQ, or in-process queue) that feeds clip writers and notification handlers.
- Surface per-camera metrics (FPS, last frame timestamp, dropped frames, notification counts) for dashboards and alerting.

### 4.9 ONVIF Integration
- Use ONVIF discovery (WS-Discovery) to auto-enumerate available cameras and populate `cameras.yml` with the correct RTSP profiles/credentials.
- Leverage an ONVIF client library (e.g., `onvif-zeep` or `python-onvif`) to pull device capabilities, current profiles, and PTZ presets when applicable.
- Schedule periodic ONVIF `GetStatus`/`GetStreamUri` checks to validate camera uptime and detect configuration drift; feed results into monitoring metrics.
- Provide optional PTZ hooks to reposition cameras when scripted patrols are needed (disabled by default but supported via the same control interface).
- Store ONVIF credentials separately (e.g., `/etc/wildlife/onvif_secrets.json`) and reference them from service units using environment files.

## 5. Implementation Phases & Checkpoints
### Phase 1 — Prototype Pipeline (Week 1)
- **Objectives**: confirm RTSP ingest reliability, YOLO inference throughput, and clip extraction on a single stream.
- **Tasks**: build minimal GStreamer → YOLO loop, log detections, write manual clip via ffmpeg, capture baseline GPU/FPS metrics.
- **Exit criteria**: ≥15 FPS sustained, <70% GPU utilization, prototype clip playable, metrics/logging enabled, ONVIF discovery returns expected RTSP profile info for test camera.
- **Tests**: replay canned RTSP sample, inject stream drop to verify reconnect, manually review detection output.

### Phase 2 — Automated Detection & Buffering (Week 2)
- **Objectives**: productionize event logic, circular buffer, and clip writer; support concurrent RTSP playback.
- **Tasks**: implement smoothing/exclusion list, wire circular buffer to encoder, schedule dual-stream test harness.
- **Exit criteria**: events generate clips with pre/post roll, buffer never underflows during dual-stream replay, SSD writes land under `/mnt/wildlife_ssd`.
- **Tests**: unit tests for state machine/thresholds, integration test with two RTSP recordings, verify SSD throughput and retention metadata.

### Phase 3 — Notification Integration (Week 3)
- **Objectives**: deliver Pushover alerts with metadata while respecting exclusion list and rate limits.
- **Tasks**: configure Pushover tokens, build notification worker, add retry/backoff + suppression for excluded species.
- **Exit criteria**: alerts reach devices within 5 s of clip finalization, exclusion list events logged but not alerted, rate limiter proven.
- **Tests**: send synthetic detections across all species, force HTTP failure to validate retries, confirm logging when alerts skipped.

### Phase 4 — Hardening & Deployment (Week 4)
- **Objectives**: package services, add observability, and ensure SSD management+watchdogs are active.
- **Tasks**: create systemd units (`detector@.service`, `ssd-cleaner.service`, `ssd-cleaner.timer`), document `/etc/wildlife/cameras.yml`, expose Prometheus metrics, add log rotation.
- **Exit criteria**: `systemctl status` clean across services, SSD cleaner enforces <80% usage, monitoring endpoint exports per-camera stats, ONVIF health checks feeding metrics/alerts.
- **Tests**: reboot test for fstab mounts, simulate SSD fill to trigger cleanup alert, run `systemd-analyze verify` on unit files.

### Phase 5 — Field Testing (Week 5)
- **Objectives**: validate behavior with live cameras and environmental conditions.
- **Tasks**: deploy near production cameras, capture telemetry, adjust thresholds per species/camera.
- **Exit criteria**: one-week run without crash, <1 false positive per day per camera, storage within budget, alerts actionable by stakeholders.
- **Tests**: review daily logs/events, spot-check clips, confirm Pushover delivery matrix.

### Phase 6 — Multi-Camera Scale-Up (Week 6+)
- **Objectives**: onboard additional streams and ensure orchestrator stability under load.
- **Tasks**: add new entries to `cameras.yml`, tune scheduler throttles, execute soak tests with synthetic bursts, validate cleanup/notification behavior during parallel events.
- **Exit criteria**: GPU utilization ≤85% with N cameras, no dropped frames >1% per stream, notification hub handles concurrent events, cleanup keeps SSD under target.
- **Tests**: chaos tests (kill worker, drop RTSP), multi-hour soak with recorded feeds, confirm dashboards reflect per-camera KPIs.

## 6. Testing Strategy
- **Unit tests**: detection thresholds, buffer window logic, notification payload builder.
- **Integration tests**: replay recorded RTSP streams to ensure detections fire correctly.
- **Performance tests**: measure FPS, latency, and GPU memory usage under peak load.
- **Failover tests**: simulate RTSP drop, storage full, Pushover outage.

## 7. Risks & Mitigations
- **RTSP instability** → implement reconnect + local caching of clips until stream resumes.
- **Model accuracy** → gather labeled data, consider transfer learning, adjust confidence per class.
- **Jetson resource limits** → offload heavy encoding (e.g., use hardware NVENC), reduce resolution.
- **Notification spam** → enforce cooldown windows and severity-based routing.

## 8. Next Steps & Deliverables
- Choose inference stack (DeepStream vs Ultralytics TensorRT) and finalize YOLO variant.
- Build proof-of-concept pipeline script and collect performance metrics.
- Define clip storage retention rules and security (file permissions, optional encryption).
- Implement SSD cleanup timer/service and document manual override commands.
- Design multi-camera configuration schema plus deployment tooling, and document operational runbooks for adding/removing streams.
- Validate ONVIF discovery/control scripts and include credential management guidance in the runbook.
- Document operational procedures (deployment scripts, troubleshooting guide).
