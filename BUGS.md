# Known issues / backlog

Open items that were diagnosed but deliberately not fixed yet, with the
evidence and the reasoning, so they can be picked up later without redoing
the investigation. Incident data referenced below lives in
`tmp/ptz_review_<timestamp>/` (gitignored; regenerate with
`scripts/ptz_review.py` per `.github/skills/ptz-incident-review/SKILL.md`).

Context: `b4abb40` and `5ad1f0c` (2026-09-07) fixed the detection starvation,
velocity throttling, false "lost" transitions, dead zoom-in, and lock
self-sabotage found in the 2026-05-09 coyote reviews. Items 1-2 below
are what is left of the tracking-quality gap after those; item 3 was found
on 2026-09-07 while re-reading the same incident journals.

---

## 1. Predictive lead compensation for PTZ tracking

**Status:** open — evaluate after a post-`5ad1f0c` baseline exists
**Area:** `src/animaltracker/ptz_tracker.py` (`_do_tracking_from_target`,
`_do_tracking`)
**Priority:** highest remaining algorithmic upgrade

### Problem

The controller is purely proportional on the subject's *last observed*
position, and that observation is stale by the time it acts on it:

| clip | `frame_age_ms` median | max |
|---|---|---|
| `ptz_review_1778332418` (coyote) | 315 | 448 |
| `ptz_review_1778509786` (reptile) | 509 | 3677 |

Add the ~1s gap between moves and each correction is aimed at where the
animal *was*, not where it will be when the slew lands. On a walking animal
the offset therefore never fully closes even with the caps and duty cycle
fixed.

### Proposed approach

Constant-velocity extrapolation in the camera's own normalized frame:

- Estimate the subject's normalized velocity from two consecutive sightings
  in the **same camera's** frame with **no PTZ move between them**
  (otherwise apparent motion is contaminated by ego-motion).
- Only trust the estimate when the two sightings are < ~1.5s apart.
- Add a lead term to the offset: `offset += v * (frame_age + 0.5 * tracking_step_duration)`,
  clamped to about ±0.20.
- When no valid estimate exists, lead = 0 and behaviour is unchanged.
- Log the lead in the `move` decision details so `scripts/ptz_review.py`
  can show when it engaged and by how much.
- Gate behind a `PTZTrackingSettings` flag (`lead_compensation_enabled`)
  so it can be turned off from config without a code change.

### Why it was deferred

- **Engagement is cadence-limited.** On cam2 every move blanks its
  detections for `ptz_settle_time` (0.5s) and cam2 infers at ~2 fps, so
  consecutive no-move sightings are rare there; the lead would mostly
  engage on cam1 (static, so its velocity estimates are clean) during
  `SOURCE_TRACKING`. Real benefit depends on the post-fix cadence, which
  has not been measured yet.
- **Cannot be validated offline.** The incident CSVs only record the
  decisions the old controller made; there is no replay harness that
  models camera motion, so the only test is a live animal.

### How to evaluate

1. With the cameras back, collect 2-3 tracking clips under `5ad1f0c` and
   run `scripts/ptz_review.py` on each to get a baseline: median
   `|offset|` per move, whether it trends toward zero, `frame_age_ms`,
   `gap_since_last_move_ms`.
2. Implement behind the flag, deploy, collect the same clips, compare.
3. Acceptance: median `|offset|` at move time drops and the
   first→last offset trend converges instead of growing; no increase in
   `tracking_lost` / `mode_change → patrol` on visible subjects.

---

## 2. GPU inference scheduling between cam1 and cam2

**Status:** open — needs measurement before any change
**Area:** `src/animaltracker/pipeline.py` (`StreamWorker.run` frame-skip
logic, `_process_frame`), `src/animaltracker/detector.py`
(`MegaDetectorBackend.infer`)
**Priority:** medium — it bounds how fast every other fix can react

### Problem

Both cameras run MegaDetector concurrently on one GTX 1080 with no
coordination. Inference is 200-250ms per frame, so cam2 — the camera that
actually matters while tracking — gets starved:

```
cam1: capture=18.6fps infer=3.1fps drop=31 (50.0%)
cam2: capture=25.8fps infer=2.2fps drop=108 (83.1%)
cam2: capture=14.9fps infer=1.7fps drop=55 (76.4%)
```

(`[PERF]` lines from `journal_0811_0816.log`.) Only 20% of cam2's raw
detection frames ever reached the PTZ tracker in that incident. Higher
cam2 cadence would directly shorten the ~1s gap between corrections.

### Candidate approaches

- **Throttle cam1 while cam2 drives.** When the shared tracker is in
  `TRACKING` and `_last_detection_source == cam2`, raise cam1's effective
  `frame_skip` (e.g. ×3) so the GPU is mostly cam2's. cam1 is only the
  fallback in that state.
- **Cheaper cam1 inference instead of fewer frames.** Set
  `inference_max_width` on cam1 (production already has it on cam2 at 960)
  to cut per-frame cost without touching cadence.
- **Serialize with priority.** A single inference queue where cam2's frame
  pre-empts cam1's when tracking; avoids two CUDA contexts contending.

### The interaction that makes this non-trivial

The multi-camera gather in `_process_frame` only accepts a camera's
published detections if they are younger than the staleness window
(`max(0.5, min(1.0, ptz_settle_time * 2))` = 1.0s in production). cam1's
detections already arrive 350-550ms old. Throttling cam1 to ~1 fps pushes
its detection age past the window and cam1 silently stops contributing —
which re-creates the "cam1 has none" failure `b4abb40` just fixed. Any
throttling must either stay above ~2 fps on cam1 or widen the window in
step (the moved camera's detections are cleared during settle anyway, so a
wider window is safe for cam2).

### How to evaluate

1. Baseline the `[PERF]` lines for both cameras during a tracking event
   under `5ad1f0c` (`infer` fps, `drop` %, `frame_age`).
2. Try the cheapest change first (`inference_max_width` on cam1 in the
   production `config/cameras.yml`; it is gitignored, edit it on the host)
   and re-measure.
3. Only then consider tracking-aware throttling, and check the journal for
   `cam1 has none` while cam1 is logging `[REALTIME]` detections — that is
   the signature of over-throttling.

---

## 3. Pre-roll clip seed blocks the shared event loop for ~5s at every event start

**Status:** fix applied 2026-09-07 (`StreamingClipWriter` now encodes on its
own thread; `_maybe_close_event` drains it in an executor) — awaiting
verification on production with a real event
**Area:** `src/animaltracker/pipeline.py` (`StreamWorker.run`, the
`clip_writer is None` branch at ~660-676), `src/animaltracker/storage.py`
(`StreamingClipWriter`)
**Priority:** highest — it is the largest single tracking gap in every
incident on file, and the per-move `frame_age_ms` metric cannot see it

### Problem

When an event opens, the stream loop constructs the `StreamingClipWriter`
and seeds it with the pre-roll buffer *synchronously*, inside
`async def run()`:

```python
for _ts, _frame in self.clip_buffer.dump():
    if _ts >= cutoff:
        self.event_state.clip_writer.write(_frame)   # cv2.VideoWriter MJPG encode
```

With `pre_seconds=10` and cam2 capturing at ~25fps that is ~250 1080p
frames through the MJPG encoder on the event-loop thread (the comment's
"this is cheap" refers to `dump()`, not the writes). Every camera worker is
`asyncio.gather`ed onto that one loop (`pipeline.py:2379`), so cam1 freezes
too: no capture, no inference, no PTZ decisions from any camera until the
seed finishes. The tracking-step auto-stop is a `threading.Timer`, so the
camera does not run away — it parks for ~5s while the animal walks.

The same inline `clip_writer.write(frame)` (`:676`) then runs for every
live frame for the rest of the event. 5s / 250 frames ≈ 20ms per write is
inferred, not measured on the host; if it is right, the loop spends a large
fraction of each second encoding during exactly the period tracking
matters. The event-close transcode, by contrast, is correctly handed to an
executor (`:1794`), so the seed is the only blocking path.

### Evidence

- `ptz_review_1778332418/journal_0811_0816.log`: "Started tracking animal
  on cam2" at 08:13:39; the next log line *of any kind* from the process is
  at 08:13:44. Both cameras silent, with the coyote in view of both (cam1
  74% at :39, 81% at :44) and cam2 having just centred it (`DEADZONE`,
  offset 0.027). It had drifted to offset 0.14 by the time the next move
  fired.
- `ptz_review_1778327229/journal_0645_0650.log`: the same ~5s hole after
  both cam2 event starts (06:45:18→:24 and 06:47:09→:15).
- `ptz_review_1778509786`: the only move with `frame_age > 1s` is 3677ms
  at t=7.7s, consistent with the event opening (that journal excerpt does
  not cover the event start, so this one is unconfirmed).
- No decisions are logged during a stall, so `frame_age_ms` per move stays
  at 300-500ms and the item-1 evaluation plan would never surface it.

### Fix (applied)

`StreamingClipWriter` owns a writer thread: `seed()` and `write()` only
queue frame references, the thread opens the cv2 writer from the first
frame and releases it on exit (one thread ever touches the handle), and
`close()` drains and joins. Seed frames are never shed; live frames are
shed — counted and logged at close — once the backlog exceeds
`max_pending` (`max(300, pre_seconds × 30)`), so capture never blocks on
a slow encoder. `_maybe_close_event` detaches the event and resets the
tracker *before* awaiting `close()` in an executor, so `run()` cannot
open a second writer for a closing event while the drain yields. Pinned by
`tests/test_streaming_clip_writer.py`. A write error is logged and
counted rather than killing the thread, so whatever was written survives
to be transcoded.

### How to evaluate

1. After the next event, run `scripts/ptz_review.py` with `--journal`; the
   new `## Journal Stalls` section reports the gap after each "Started
   tracking" line. On the May journals it reads `+5s`, `+6s`, `+6s` for
   cam2 starts and `+2s` for a cam1 start; it should read `+0s`/`+1s`.
2. `frame_age_ms` of the first move after an event opens stays under
   ~500ms.
3. cam2 `[PERF]` `drop` % during events vs. the item-2 baseline; it should
   fall if the per-frame write was loading the loop.
4. Saved clips still contain `pre_seconds` of context and play correctly.

---

## 4. Post-processing runs at ~1 sampled frame/s with three cameras

**Status:** open — found 2026-09-14 while tracing "No frame" cards
**Area:** `src/animaltracker/postprocess.py` (`_analyze_video`),
`src/animaltracker/detector.py` (SpeciesNet per-box classification)

### Problem

With cam1 plus the two 2688x1512 Otteson cameras the GTX 1080 sits at ~95%
on real-time MegaDetector alone, and SpeciesNet post-processing gets what
is left: the 20 s cam1 clip `1789385669_animal.mp4` took 06:34:55 →
06:39:58 for 305 sampled frames (673 boxes classified one by one), i.e.
about one sampled frame per second, against 2–4 frames/s measured on
2026-09-09 with cam1 alone. A 6-minute Otteson2 clip (5435 frames,
sample_rate 2) needs ~45 min, during which the archive card shows the
clip as analysing and a restart loses the whole job (now recovered by
`analysis_recovery.py`, but the wait remains). Real-time inference also
slows from ~170 ms to ~330 ms per frame while a job runs.

### Options

- Sub-streams for the Otteson cameras (also fixes the per-event "encoder
  fell behind capture" drops): far fewer real-time pixels on the GPU.
- Batch the per-box classifier calls per frame instead of one call per box.
- A larger `sample_rate` for long clips (post_analysis_frames already
  auto-scales; the real cost is the per-box classification).

---

## 5. SpeciesNet load races real-time inference through a torch.fx trace (fixed 2026-09-21)

**Status:** fixed — `ModelLoadGate` in `detector.py`. Forward passes hold
the gate shared (through `BaseDetector.model_lock`, which every backend's
forward already ran under), a model build holds it alone, and a waiting
build blocks new forwards so three cameras cannot starve it. The weights
are resolved (`ModelInfo`, the download) before the gate is taken. The
race is reproduced with plain torch in `tests/test_model_load_gate.py`.
Cost: live detection pauses for the length of the SpeciesNet load, a few
seconds, once per process, in place of one lost frame and a traceback.
First seen 2026-09-14 06:34:55 on the first clip after a restart
**Area:** `src/animaltracker/pipeline.py` (`_get_postprocess_detector`),
`src/animaltracker/detector.py`

### Problem

Loading the post-process SpeciesNet (lazily, on the first clip after each
start) runs a `torch.fx` symbolic trace, which patches
`torch.nn.Module.__call__` process-wide for its duration. A real-time
MegaDetector forward on another thread during that window raises
`NameError: module is not installed as a submodule` from
`torch/fx/_symbolic_trace.py` (`path_of_module`); the journal shows it as
`ERROR animaltracker.pipeline: Inference error for Otteson2` with a
traceback. One frame is lost and the worker continues, so the cost is one
scary log line per restart — but it will recur on every first clip, and
the recovery sweep now triggers that load ~90 s after every start.

### Options

- Load the post-process detector before the camera workers start (no
  inference is running yet), accepting ~2 s more startup and the VRAM
  up front; or
- a process-wide read/write lock: inference takes the read side, model
  loading the write side.

---

## 6. PTZ defects that cannot run in production today

**Status:** open, dormant — recorded 2026-09-21 at the operator's request
**Area:** `src/animaltracker/ptz_tracker.py`, `ptz_calibration.py`,
`onvif_client.py`, `web.py`, `pipeline.py`

No camera in production has an `onvif` block, `ptz_tracking.enabled` is off
on all three, and `cam2`, the only zoom camera, is retired. None of the
following can happen until a PTZ camera comes back; fix them then, and
test on the hardware, because none can be proved without it. Found in the
2026-09-19 bug hunt. The first five were re-checked against the code on
2026-09-21; line numbers are as of `7d24274`.

- **Investigate mode can never start.** It points cam2 at a detection too
  small to track, one between `investigate_min_area` and
  `min_detection_area`, and `PTZTracker` collects those "before size
  filtering removes them" (`ptz_tracker.py:1623`). But the pipeline hands
  the tracker its detections after `StreamWorker._filter_false_positives`
  has applied `min_detection_area` (`pipeline.py:1435`, then the `update` /
  `update_multi_camera` calls below it), so the candidates never arrive.
  Fixing it makes cam2 move more, which is why it wants a decision.
- **Investigate mode gives up on the first empty tick.** In
  `INVESTIGATE`, any update without a candidate calls
  `_maybe_finish_investigate(now, confirmed=False)` (`ptz_tracker.py:1950`)
  with no check of `investigate_timeout`, logs "[INVESTIGATE_TIMEOUT] ...
  within 4.0s" however long it waited, and puts the spot on the 30 s
  cooldown. One frame where the small detection flickers out ends it.
- **Zoom-FOV bounds are inflated when the wide frame was upscaled.**
  `ZoomFOVCalibrator` upscales the wide frame to the zoom frame's size for
  matching (`ptz_calibration.py:705`) but normalises the matched corners
  by the original wide size (`:806-809`), so with cam1 on a sub-stream and
  cam2 on the main stream every calibrated box is `scale_up` times too
  big. Visibility recovery reads that calibration.
- **The web UI saves the calibration by coincidence of layout.**
  `web.py:1038` writes `<storage_root>/../config/zoom_fov_calibration.json`;
  the tracker reads `ptz_tracking.zoom_fov_calibration_path` (default
  `config/zoom_fov_calibration.json`, relative to the working directory).
  They are the same file only because production's storage lives inside
  the checkout; with storage on its own mount the tracker never sees a
  calibration run from the UI. Save to the configured path.
- **The ONVIF timeout is applied after the constructor's network calls.**
  `OnvifClient.__init__` builds `ONVIFCamera(...)`, which talks to the
  camera (capabilities, service addresses), and only then calls
  `_apply_transport_timeout()` (`onvif_client.py:56-61`). A camera that
  accepts the connection and never answers hangs worker start-up with no
  timeout.
- **Visibility recovery** (`_do_visibility_recovery_from_source`,
  `ptz_tracker.py:919`) was reported to never put the tracker in
  `TRACKING` after its move, and every branch sets a non-zero zoom
  velocity, so it zooms (in or out) on every call even with the target
  centred and a good size. Not re-checked on 2026-09-21.
- **The single-camera path was reported never to get the wide post-slew
  lock radius** (`_lock_spatial_radius_after_own_move`, chosen at
  `ptz_tracker.py:2831-2838` only when the anchor's source is the PTZ
  camera itself). Not re-checked on 2026-09-21.

---

## Awaiting a decision (not a tracking item)

### Storage retention (fixed 2026-09-20)

`cleanup()` globbed `clips/*/*/*/*`, one level short of
`clips/<cam>/<YYYY>/<MM>/<DD>/<file>`, so every match was a directory and
nothing was ever deleted. `StorageManager.prune_clips` replaces it:

- ages a clip by the event epoch in its name, not its mtime (a reanalysis
  rewrites the sidecar and leaves the video alone, so a clip's files
  disagree about their own age);
- removes a clip with its key frames and its log;
- treats `retention.min_days` as a floor nothing overrides, disk pressure
  included, and honours `max_utilization_pct` after the age pass, oldest
  first, stopping when nothing more can go;
- sweeps a `*.temp.avi` or `*.tmp.mp4` with no MP4 beside it once it is past
  `max_days`, logged for what it is; a recent one is kept and reported,
  because it may still be an event worth recovering or a transcode in
  flight. Intermediates are aged by mtime, never by the epoch in their name:
  the recovery sweep transcodes months-old events, and dating one by its
  event would let a concurrent pass delete it mid-write.

`cleanup --dry-run` prints what would go (counts, GB, date span, per camera)
and the command's exit status is now meaningful, so `ssd-cleaner.service`
can see a failure.

The three interrupted recordings on production (3.7 GB, all `cam2`) were
deleted by hand on 2026-09-20 at the operator's request; the archive holds
none now.

The operator ran the first real pass on production on 2026-09-20: 247 of
438 clips (24 Dec 2025 to 21 May 2026, 175 of them from the retired `cam2`),
exactly the files the dry run listed, checked afterwards against a copy of
the archive taken before it.

Then, 2026-09-21:

- key frames and logs whose clip is already gone go too, once past
  `max_days` (`StorageManager.find_leftovers`). A delete that removed only
  the video left them (the web delete did until e236509), and nothing found
  them because the archive and retention start from the videos: 954 events,
  4,370 files, 0.8 GB on production. Grouped by directory and event epoch,
  never by name; a group with any video in it, finished or half-written, or
  whose event is still recording to `event_temp`, or holding a file the
  pipeline does not write, is left alone.
- `ssd-cleaner.service` ran `cleanup --config ...`, which argparse rejects
  (`--config` belongs before the subcommand), so the unit would have failed
  on every run; it had never been installed. Fixed, the timer is daily with
  `Persistent=true`, and `tests/test_systemd_units.py` parses every unit's
  command with the real parser. Production had no timer at all, so
  `max_days` held only when someone ran the command.
