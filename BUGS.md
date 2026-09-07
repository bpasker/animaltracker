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

## Awaiting a decision (not a tracking item)

### Storage retention has never deleted anything

`src/animaltracker/storage.py` `cleanup()` and `get_clips_sorted_by_age()`
glob `clips/*/*/*/*`, one level shallower than the real layout
`clips/<cam>/<YYYY>/<MM>/<DD>/<file>`, so the hourly `ssd-cleaner` timer
has deleted zero files since the initial commit and `max_days` is not
enforced. Fixing the glob also arms `ensure_space_for_clip()`, whose
whole-filesystem utilization check can never be satisfied by deleting
clips when non-clip data already exceeds `max_utilization_pct` — it then
deletes every clip and still fails. Fix both together, bound deletion by
`retention.min_days`, and stop when a pass frees nothing.

Held back because the first run after the fix will delete everything older
than `max_days` (120 on production) in one go — that needs an explicit
go-ahead, not a ride-along in a tracking deploy.
