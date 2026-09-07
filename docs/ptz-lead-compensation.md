# Lead compensation: what NOLO does, what to take, and what the data says

Reference notes for [BUGS.md](../BUGS.md) item 1 ("Predictive lead
compensation for PTZ tracking"), written after reading
[doxx/NOLO](https://github.com/doxx/NOLO) — a Go PTZ tracker that has run a
24/7 public livestream on a Hikvision PTZ since Aug 2025 — and then checked
against the three incident reviews in `tmp/ptz_review_*`.

Line references are against `doxx/NOLO@main` (`tracking/spatial_integration.go`,
`ptz/camera_state.go`). NOLO is under a custom non-commercial "NOLO Ethical
AI License" that explicitly permits wildlife monitoring, but this is a
notes-from-reading document — **port the ideas, do not copy the code**.

**Summary of the review (section 7):** the lead is well-founded but
second-order for the incidents on file; the first draft of this document
recommended a polled camera-state manager as a prerequisite and that
recommendation is withdrawn; the incident data instead points at a ~5s
process-wide stall at every event start, caused by the pre-roll clip seed
running synchronously on the shared asyncio loop.

---

## 1. The architectural difference that changes everything

NOLO and this project actuate the camera in fundamentally different ways,
and every design decision below follows from that.

| | NOLO | animaltracker |
|---|---|---|
| Actuation | ONVIF/Hikvision `absolutePosition` to a computed Pan/Tilt/Zoom | ONVIF `ContinuousMove` velocity + timed auto-`Stop` |
| Move duration | until arrival (polled, 800ms tick) | `tracking_step_duration` = 0.35s, then a timer fires `ptz_stop` |
| Lead is expressed as | a **position offset** added to the target coordinate | must be added to `offset_x`/`offset_y` *before* `_velocity_curve` |
| Knows if the camera is moving | yes — `CameraStateManager` polls actual position | approximated — `is_settling()` is a timer since `_last_move_time` |

NOLO computes an absolute destination, so adding `velocity × leadTime` to it
is trivially correct. We compute a normalized offset and map it through
`_velocity_curve()` into a velocity that runs for a fixed pulse. So the lead
has to go in earlier in the chain, which is what BUGS.md already proposes:

```python
offset_x += lead_x   # before _velocity_curve()
offset_y += lead_y
```

That part of the plan is right.

## 2. PTZ-space does **not** solve ego-motion

NOLO's velocity estimator gates on camera-idle exactly the way BUGS.md
proposed to (`calculateBoatVelocity`, `spatial_integration.go:1892`):

```go
// SIMPLE APPROACH: Only calculate velocity when camera is IDLE
if si.cameraStateManager != nil && !si.cameraStateManager.IsIdle() {
    return
}
```

What PTZ-space buys is **zoom independence**. A lead of N PTZ pan units
means the same physical rotation at Z10 and at Z120, whereas a lead
expressed in normalized frame units means a very different rotation
depending on the current FOV (`spatial_integration.go:4058`):

```go
// === PTZ-SPACE LEAD TRACKING ===
ptzVelPan  := velX / panPixelsPerUnit
ptzVelTilt := velY / tiltPixelsPerUnit
leadPan  := ptzVelPan  * leadTime
leadTilt := ptzVelTilt * leadTime
target.Pan  += leadPan + biasPan
target.Tilt += leadTilt
```

For us that matters because cam2's zoom varies continuously under
`_zoom_velocity_for_fill()`. A lead clamped to ±0.20 in normalized cam2
coordinates is a much bigger real movement at high zoom than at low zoom.
Convert through `ZoomFOVCalibration` before clamping, or make the clamp
zoom-scaled.

NOLO also keeps a cruder second line of defence against ego-motion leaking
into the estimate — a hard speed cap (`spatial_integration.go:1926`,
250 px/s, "still filters 2000+ camera artifacts"). Cheap and worth copying:
a real animal has a plausible maximum apparent speed; anything above it is
ego-motion or an ID swap.

## 3. The idle gate is affordable for NOLO, not for us

BUGS.md deferred partly because "consecutive no-move sightings are rare" on
cam2. NOLO shows why that is a *them* problem rather than an *us* problem:

- NOLO captures at 30fps and infers YOLOv8n in ~13ms on GPU. Between two
  absolute moves it still collects many idle frames. It estimates velocity
  over the last 5 pixel-history points and *assumes 30fps* for the time
  base (`frameTimeDiff := float64(historyCount-1) / 30.0`) — it never even
  timestamps them.
- cam2 here infers at ~2fps and blanks for `ptz_settle_time` (0.5s) after
  every move. Under the same gate cam2 would produce a usable estimate
  almost never.

**So the lead should be estimated on cam1, not cam2.** cam1 is static: it is
always "idle", its estimates are never contaminated, and it sees the animal
continuously while cam2 is settling (~3 fps in the incident perf lines, so
two consecutive sightings are ~330ms apart — inside BUGS.md's 1.5s trust
window). That means the lead belongs in `_do_tracking()`
(`src/animaltracker/ptz_tracker.py:2766`, the cam1→cam2 path), and
`_do_tracking_from_target()` (`:1874`) should consume a cam1-derived lead
rather than computing its own. Use `_current_capture_ts` deltas, not an
assumed frame rate.

## 4. Withdrawn: a polled IDLE/MOVING camera state as a prerequisite

The first draft of this document recommended porting NOLO's
`CameraStateManager` (`ptz/camera_state.go:448-620`) — a position-polling
IDLE/MOVING state machine with arrival detection — as step 1, on the basis
that `is_settling()` (`ptz_tracker.py:1035`) is "a pure timer, wrong in both
directions". Checked against the code and data, that does not hold:

- **There is no target to detect arrival against.** NOLO's `checkArrival`
  compares actual position to the commanded absolute target. A
  `ContinuousMove` has no target position; the only thing a poll could
  detect is "position unchanged between two polls", which with an 800ms
  poll and a 350ms pulse is the timer again, slower and noisier.
- **The timer already models the pulse well.** Every `tracking_step_stop`
  in the three incidents fired within 1–2ms of schedule
  (`dispatch_late_ms`), and the `actual_slew_ms` column is flat at the
  configured step. There are no long slews in tracking mode to mis-model.
- **The poll is not free and its cost is unmeasured.** `ptz_get_position()`
  takes `_call_lock` (`onvif_client.py:288`), the same lock every move and
  stop takes. Not one `GetStatus` appears in any incident journal, so there
  is no evidence of its latency on this camera. Recommending a mechanism
  that contends with move dispatch, without knowing its cost, was wrong.

What survives from `camera_state.go` is the external-takeover detector
(`checkExternalMovement`, `:466`) — nice-to-have, unrelated to tracking
quality — and the idea of *logging* PTZ position with each decision so
`scripts/ptz_review.py` can subtract camera motion offline. Neither is a
prerequisite for the lead.

## 5. Smaller mechanisms worth stealing

**Edge bias** (`spatial_integration.go:2027`) — for the first 1–3 detections
of a new track there is no velocity estimate yet, so NOLO guesses direction
from position: a target in the outer third of the frame is assumed to be
heading toward the centre, and gets a fixed 20-PTZ-unit lead that way. It
degrades to zero as soon as a real estimate exists.

**Coast / recovery on loss** (`prepareRecoveryData` `:4266`,
`executePredictiveMove1` `:4376`) — on losing the target, NOLO snapshots the
PTZ-space velocity at the moment of loss and extrapolates from it, zooming
out 20% to widen the search, with per-move clamps. Our
`_do_visibility_recovery_from_source()` (`ptz_tracker.py:875`) already does
the smarter *zoom-out* half using `ZoomFOVCalibration`; what it lacks is
the extrapolation. Same missing ingredient as the lead.

**Target identity in PTZ space.** NOLO's "spatial tracker" keeps each
target's position in pan/tilt coordinates, so a camera move does not make
the target appear to jump. We hit exactly that failure in the May 9
incident (`LOCK_HOLD ×3 → LOCK_RELEASE → re-lock → DEADZONE` on a coyote a
single pulse had just centred); it is already patched by
`_lock_spatial_radius_after_own_move = 0.50` (`ptz_tracker.py:458`). The
PTZ-space anchor is the principled version of that patch, if the patch
proves insufficient.

## 6. What not to copy

- `pipelineLatency = 2.0` is hardcoded at `spatial_integration.go:526` and
  compensated in `predictPTZMovement` (`:3808`) *and again* as `leadTime`
  in `smartPTZTracking` (`:4086`). We measure the real number per decision
  (`frame_age_ms`) — use the measurement, apply it once.
- Magic gates: `if speed > 5.0` (px/s) appears twice with no explanation.
- The source shows prediction added, ripped out ("like July 6th version"),
  and re-added more than once. Gate ours behind `lead_compensation_enabled`
  and log the applied lead in the `move` decision so `ptz_review.py` can
  prove whether it helps.
- NOLO has **zero** test files across 896KB of Go.

## 7. What the incident data actually says

The three reviews in `tmp/ptz_review_*` all pre-date the Sep 7 fixes
(`b4abb40`, `5ad1f0c`). On May 9 the defaults in effect were
`low_fill_velocity_cap=0.15`, `tracking_step_duration=0.20`,
`zoom_in_offset_gate=0.05` (commit `d95d793`); today they are 0.30 / 0.35 /
0.15. Read with that in mind:

**The offset failed to close because the controller was throttled, not
because it was aiming at a stale position.** `cap_active` is true on 22/23
moves in `ptz_review_1778332418` and 9/9 in `ptz_review_1778327229`. Five
consecutive cam1-driven moves at t=0.7–1.8s held offset_y at +0.45…+0.50
while the commanded tilt velocity sat pinned at 0.15. Later, with the
subject at 20–28% fill and offset ~0.12–0.18, the cap scaled the command
down to ~0.05 (`0.15 × 0.15/0.40`), and five pulses in a row moved the
offset not at all. A lead term is added *before* the cap and would have
been capped away with the rest.

**When a pulse was allowed to be large enough, it worked in one shot.** At
08:13:37 a single 200ms pulse at velocity (0.07, −0.13) took the coyote from
offset (+0.20, −0.34) to (−0.03, +0.00). The gain is fine.

**The lead is second-order for these clips.** Between the DEADZONE at
08:13:39 and the next move at 08:13:44 the coyote drifted 0.14 of cam2's
frame in ~5s — about 0.03 frame/s. Against BUGS.md's own
`frame_age + 0.5 × step` ≈ 0.45s, the lead would be ~0.013. The residual
offsets it was meant to close were 0.12–0.50.

**The biggest gap in tracking is a process stall, and the `frame_age`
metric cannot see it.** In `journal_0811_0816.log`, "Started tracking
animal on cam2" is logged at 08:13:39 and the *next line of any kind* from
the whole process is at 08:13:44 — cam1 and cam2 both silent, with the
coyote in view of both (cam1 74% at :39, 81% at :44). The other coyote
journal shows the same 5s hole after both of its cam2 event starts
(06:45:18→:24 and 06:47:09→:15). The reptile review's only move with
`frame_age > 1s` is 3677ms at t=7.7s, right where its event would have
opened. No decisions are logged during a stall, so the per-move
`frame_age_ms` statistic stays at 300–500ms and BUGS.md's evaluation plan
would never surface it.

The cause is in the stream loop, `pipeline.py:660-673`: when an event
opens, the loop constructs the `StreamingClipWriter` and seeds it with the
pre-roll buffer —

```python
for _ts, _frame in self.clip_buffer.dump():
    if _ts >= cutoff:
        self.event_state.clip_writer.write(_frame)   # cv2.VideoWriter MJPG encode
```

— synchronously, inside `async def run()`. With `pre_seconds=10` and cam2
capturing at ~25fps that is ~250 1080p frames through the MJPG encoder on
the event-loop thread (the comment's "this is cheap" is about the `dump()`,
not the writes). Every camera worker is `asyncio.gather`ed onto that one
loop (`pipeline.py:2379`), so cam1 freezes too. 5s / 250 frames ≈ 20ms per
write, which the same inline `clip_writer.write(frame)` at `:676` then pays
for every live frame for the rest of the event. The event-close transcode,
by contrast, is correctly handed to an executor (`:1794`). The tracking-step
auto-stop is a `threading.Timer`, so the camera does not run away during the
stall — it just parks for five seconds while the animal walks.

## 8. Revised order

1. **Move `StreamingClipWriter` off the event loop** — done 2026-09-07
   (BUGS.md item 3 has the details). Verify on the next production event
   with `scripts/ptz_review.py --journal`: the `## Journal Stalls` section
   should show `+0s`/`+1s` after "Started tracking" (the May journals show
   `+5s`/`+6s`), and `frame_age_ms` on the first move after the event
   opens should stay under ~500ms.
2. Baseline against `5ad1f0c` per the BUGS.md plan — but add two columns
   to `ptz_review.py`: the largest gap between consecutive cam2 inference
   ticks during the event, and whether `cap_active` still pins the command
   once fill rises. Watch specifically for the subject parking at offset
   ~0.15: zoom-in is gated at `zoom_in_offset_gate = 0.15` and the cap-bound
   residual in the old data sat at 0.12–0.18, right on that gate.
3. If the baseline still shows a lag-shaped residual (offset sign tracking
   direction of travel, `gap_capture_to_capture` short): velocity
   estimation on cam1 from `_current_capture_ts`, plausible-max-speed cap,
   logged but unused.
4. Lead term into `_do_tracking()` behind `lead_compensation_enabled`,
   converted through `ZoomFOVCalibration`.
5. Edge bias for the first 1–3 detections.
6. Velocity extrapolation inside the existing visibility-recovery path.
