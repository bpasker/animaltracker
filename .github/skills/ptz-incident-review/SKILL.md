---
name: ptz-incident-review
description: 'Use when: investigating PTZ tracking failures, overshoot, stale detections, missed centering, cam1/cam2 handoff problems, animal clips, sidecar .log.json files, journal logs, or requests to review why a recording did not keep an animal centered.'
argument-hint: 'clip URL/path, camera id, timestamp, or symptoms'
---

# PTZ Incident Review

Use this workflow when a user asks why PTZ tracking missed, overshot, failed to center, lost an animal, or behaved oddly in a saved clip.

## Goals

- Correlate video, clip sidecar metadata, PTZ decisions, and service logs.
- Explain the behavioral cause in concrete terms: stale frames, duplicate moves, late stops, handoff, calibration, detection latency, bad boxes, or config.
- Produce reusable artifacts the user can inspect.
- Fix the root cause when code or config changes are warranted.

## Procedure

1. Identify the clip timestamp, camera id, and approximate local time from the URL or file path.
2. Collect the MP4, matching `.log.json` sidecar, and a `journalctl -u animaltracker` slice covering at least two minutes before and after the event.
3. Put temporary evidence under `tmp/ptz_review_<timestamp>/`; do not commit those artifacts.
4. Run the repo review tool:

   ```bash
   ./.venv/bin/python scripts/ptz_review.py \
     --clip tmp/ptz_review_<timestamp>/<clip>.mp4 \
     --log-json tmp/ptz_review_<timestamp>/<clip>.log.json \
     --journal tmp/ptz_review_<timestamp>/journal.log \
     --out-dir tmp/ptz_review_<timestamp>/review
   ```

5. Read `ptz_review_report.md`, inspect the generated `ptz_keyframes.jpg`, and correlate with PTZ code in `src/animaltracker/ptz_tracker.py` and multi-camera dispatch in `src/animaltracker/pipeline.py`.
6. Look specifically for:
   - duplicate same-frame moves (`gap_capture_to_capture_ms` near `0`)
   - `tracking_step_stop` dispatch lateness
   - high `frame_age_ms` on moves
   - holes in journal output right after `Started tracking` (`## Journal Stalls` in the report): a process-wide stall that the per-move `frame_age_ms` cannot see, because no decisions are logged during it
   - tiny target fill with high velocity
   - oversized or impossible detection boxes
   - immediate cam1 fallback after cam2 had the target
   - target-camera takeover or continuity rejection issues
7. If changing `src/`, `config/`, or `systemd/`, follow the deployment workflow in `AGENTS.md`: commit, push, pull on the production host, and restart `animaltracker`.
8. Validate with syntax checks, focused fake ONVIF checks when possible, and remote `journalctl` after restart.

## Reporting

In the final response, include:

- what went wrong, grounded in the report metrics and visual review
- what artifacts were generated and where
- what code/config changed, if anything
- validation and deployment status
