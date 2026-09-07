#!/usr/bin/env python3
"""Review a PTZ tracking clip using its post-process sidecar and optional journal.

Produces:
- ptz_review_report.md: human-readable diagnosis summary
- ptz_centers.csv: per-detection normalized center/fill timeline
- ptz_decisions.csv: saved PTZ decision timeline
- ptz_keyframes.jpg: annotated contact sheet of representative detections
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Iterable

import cv2


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return ""
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _bbox_center(row: dict[str, Any], frame_width: float, frame_height: float) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(v) for v in row["bbox"]]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = max(x2 - x1, 0.0)
    height = max(y2 - y1, 0.0)
    return {
        "frame_idx": int(row.get("frame_idx", -1)),
        "event": row.get("event", ""),
        "species": row.get("species", ""),
        "confidence": float(row.get("confidence") or 0.0),
        "track_id": row.get("track_id"),
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "center_x_px": center_x,
        "center_y_px": center_y,
        "center_x_norm": center_x / frame_width if frame_width else 0.0,
        "center_y_norm": center_y / frame_height if frame_height else 0.0,
        "offset_x": (center_x / frame_width - 0.5) if frame_width else 0.0,
        "offset_y": (0.5 - center_y / frame_height) if frame_height else 0.0,
        "fill_pct": max(width / frame_width, height / frame_height) * 100.0 if frame_width and frame_height else 0.0,
    }


def detection_rows(log: dict[str, Any], frame_width: float, frame_height: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in log.get("log_entries", []):
        if entry.get("event") not in {"accepted", "tracked"}:
            continue
        if not entry.get("bbox") or int(entry.get("frame_idx", -1)) < 0:
            continue
        rows.append(_bbox_center(entry, frame_width, frame_height))
    return rows


def representative_rows(rows: list[dict[str, Any]], count: int = 12) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    indexes = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    return [rows[i] for i in indexes]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_decisions(log: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision in log.get("ptz_decisions", []):
        details = decision.get("details") or {}
        velocity = details.get("velocity") or {}
        velocity_raw = details.get("velocity_raw") or {}
        offset = details.get("offset") or {}
        rows.append({
            "timestamp": decision.get("timestamp"),
            "time": _fmt_ts(decision.get("timestamp")),
            "event": decision.get("event"),
            "mode": decision.get("mode"),
            "source": details.get("source") or details.get("source_camera"),
            "species": details.get("species"),
            "track_id": details.get("track_id"),
            "confidence": details.get("confidence"),
            "bbox_px": details.get("bbox_px"),
            "offset_x": offset.get("x"),
            "offset_y": offset.get("y"),
            "offset_magnitude": offset.get("magnitude"),
            "pan": velocity.get("pan"),
            "tilt": velocity.get("tilt"),
            "zoom": velocity.get("zoom"),
            "pan_raw": velocity_raw.get("pan"),
            "tilt_raw": velocity_raw.get("tilt"),
            "cap_active": details.get("cap_active"),
            "fill_pct": details.get("fill_pct"),
            "frame_age_ms": details.get("frame_age_ms"),
            "gap_since_last_move_ms": details.get("gap_since_last_move_ms"),
            "gap_capture_to_capture_ms": details.get("gap_capture_to_capture_ms"),
            "actual_slew_ms": details.get("actual_slew_ms"),
            "dispatch_late_ms": details.get("dispatch_late_ms"),
        })
    return rows


def render_contact_sheet(video_path: Path, rows: list[dict[str, Any]], output_path: Path) -> bool:
    if not rows:
        return False
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False
    thumbs = []
    for row in representative_rows(rows):
        frame_idx = int(row["frame_idx"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, int(round(row["bbox_x1"]))))
        y1 = max(0, min(height - 1, int(round(row["bbox_y1"]))))
        x2 = max(0, min(width - 1, int(round(row["bbox_x2"]))))
        y2 = max(0, min(height - 1, int(round(row["bbox_y2"]))))
        center_x = int(round(row["center_x_px"]))
        center_y = int(round(row["center_y_px"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
        cv2.drawMarker(frame, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 28, 3)
        cv2.drawMarker(frame, (width // 2, height // 2), (0, 255, 0), cv2.MARKER_CROSS, 40, 3)
        label = (
            f"f{frame_idx} t={frame_idx / max(capture.get(cv2.CAP_PROP_FPS), 1):.1f}s "
            f"off=({row['offset_x']:+.2f},{row['offset_y']:+.2f})"
        )
        cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        thumbs.append(cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return False
    while len(thumbs) % 3:
        thumbs.append(255 * thumbs[-1])
    sheet_rows = [cv2.hconcat(thumbs[i:i + 3]) for i in range(0, len(thumbs), 3)]
    cv2.imwrite(str(output_path), cv2.vconcat(sheet_rows))
    return True


def journal_perf_summary(path: Path | None) -> list[str]:
    if not path or not path.exists():
        return []
    perf_lines: list[str] = []
    pattern = re.compile(r"\[PERF\] (?P<camera>\w+): (?P<details>.*)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            perf_lines.append(f"{match.group('camera')}: {match.group('details')}")
    return perf_lines[-6:]


_JOURNAL_TS = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2}) +(?P<day>\d{1,2}) (?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}) "
)
_JOURNAL_PREFIX = re.compile(r"^\S+ \S+\[\d+\]: (?:[A-Z]+:[\w.]+:)?")


def journal_stall_summary(path: Path | None, min_gap_s: int = 2) -> list[str]:
    """Holes in the journal's own output.

    A live pipeline logs several lines a second while an animal is in view,
    so a multi-second hole in *all* output is a process-wide stall (BUGS.md
    item 3: the pre-roll clip seed blocking the shared event loop). The
    per-move frame_age statistic cannot see it because no decisions are
    logged during it; this is the check that can. Reports the gap after
    each "Started tracking" line -- where the seed runs -- and every hole of
    at least ``min_gap_s`` with the line that preceded it. Idle stretches
    before or after the event show up here too; the ones that matter are
    inside it. journalctl timestamps are whole seconds.
    """
    if not path or not path.exists():
        return []
    entries: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"GET /' in line:
            continue  # web access noise
        match = _JOURNAL_TS.match(line)
        if not match:
            continue
        secs = int(match["day"]) * 86400 + int(match["h"]) * 3600 + int(match["m"]) * 60 + int(match["s"])
        entries.append((secs, _JOURNAL_PREFIX.sub("", line[match.end():]).strip()))
    if not entries:
        return []

    def _clock(secs: int) -> str:
        secs %= 86400
        return f"{secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}"

    lines: list[str] = []
    for idx, (secs, text) in enumerate(entries):
        if "Started tracking" not in text:
            continue
        following = next((t for t, _ in entries[idx + 1:] if t > secs), None)
        gap = f"+{following - secs}s" if following is not None else "n/a"
        lines.append(f"- event start {_clock(secs)}: {text[:70]} -> next log line {gap}")
    holes = [
        (t1 - t0, t0, t1, text0)
        for (t0, text0), (t1, _) in zip(entries, entries[1:])
        if t1 - t0 >= min_gap_s
    ]
    if holes:
        for gap, t0, t1, text0 in sorted(holes, key=lambda h: -h[0])[:10]:
            lines.append(f"- {gap}s hole {_clock(t0)} -> {_clock(t1)} after: {text0[:80]}")
    else:
        lines.append(f"- no holes >= {min_gap_s}s in journal output")
    return lines


def build_report(
    log: dict[str, Any],
    center_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    perf_lines: list[str],
    output_dir: Path,
    stall_lines: list[str] | None = None,
) -> str:
    video = log.get("video", {})
    summary = log.get("analysis_summary", {})
    tracks = log.get("tracking_summary", {}).get("tracks", [])
    moves = [row for row in decision_rows if row.get("event") == "move"]
    stops = [row for row in decision_rows if row.get("event") == "tracking_step_stop"]
    duplicate_moves = [
        row for row in moves
        if row.get("gap_capture_to_capture_ms") is not None and abs(float(row["gap_capture_to_capture_ms"])) < 1e-6
    ]
    duplicate_skips = [row for row in decision_rows if row.get("event") == "duplicate_frame_skip"]
    late_stops = [row for row in stops if (row.get("dispatch_late_ms") or 0) > 100]
    frame_ages = [float(row["frame_age_ms"]) for row in moves if row.get("frame_age_ms") is not None]

    lines = [
        "# PTZ Review",
        "",
        f"Clip: `{log.get('clip', '')}`",
        f"Duration: {video.get('duration_seconds', 0):.1f}s, frames: {video.get('total_frames', 0)}, fps: {video.get('fps', 0)}",
        f"Post-process detections: {summary.get('frames_with_detections', 0)} / {summary.get('frames_analyzed', 0)} analyzed frames",
        "",
        "## Tracks",
    ]
    for track in tracks:
        lines.append(
            f"- track {track.get('track_id')}: {track.get('best_species')} "
            f"conf={track.get('best_confidence')} frames={track.get('first_frame')}-{track.get('last_frame')} "
            f"seen={track.get('frames_seen')}"
        )

    lines.extend(["", "## Centering"])
    if center_rows:
        first = center_rows[0]
        last = center_rows[-1]
        max_abs_x = max(center_rows, key=lambda row: abs(row["offset_x"]))
        max_abs_y = max(center_rows, key=lambda row: abs(row["offset_y"]))
        lines.extend([
            f"- first detection frame {first['frame_idx']}: offset=({first['offset_x']:+.3f}, {first['offset_y']:+.3f}), fill={first['fill_pct']:.1f}%",
            f"- last detection frame {last['frame_idx']}: offset=({last['offset_x']:+.3f}, {last['offset_y']:+.3f}), fill={last['fill_pct']:.1f}%",
            f"- worst horizontal offset frame {max_abs_x['frame_idx']}: {max_abs_x['offset_x']:+.3f}",
            f"- worst vertical offset frame {max_abs_y['frame_idx']}: {max_abs_y['offset_y']:+.3f}",
        ])
    else:
        lines.append("- no accepted/tracked detection boxes found in sidecar")

    lines.extend(["", "## PTZ Decisions"])
    lines.extend([
        f"- moves: {len(moves)}",
        f"- duplicate same-frame moves: {len(duplicate_moves)}",
        f"- duplicate same-frame skips logged: {len(duplicate_skips)}",
        f"- tracking step stops: {len(stops)}",
        f"- late step stops >100ms: {len(late_stops)}",
    ])
    if frame_ages:
        lines.append(f"- move frame age avg/max: {sum(frame_ages) / len(frame_ages):.0f}ms / {max(frame_ages):.0f}ms")
    if duplicate_moves:
        lines.append("- duplicate move examples:")
        for row in duplicate_moves[:5]:
            lines.append(f"  - {row['time']} bbox={row.get('bbox_px')} velocity=({row.get('pan')}, {row.get('tilt')})")
    if late_stops:
        lines.append("- late stop examples:")
        for row in late_stops[:5]:
            lines.append(f"  - {row['time']} actual={row.get('actual_slew_ms')}ms late={row.get('dispatch_late_ms')}ms")

    if perf_lines:
        lines.extend(["", "## Recent Perf Lines"])
        lines.extend(f"- {line}" for line in perf_lines)
    if stall_lines:
        lines.extend(["", "## Journal Stalls"])
        lines.extend(stall_lines)

    lines.extend([
        "", "## Outputs",
        f"- centers CSV: `{(output_dir / 'ptz_centers.csv').name}`",
        f"- decisions CSV: `{(output_dir / 'ptz_decisions.csv').name}`",
        f"- keyframes: `{(output_dir / 'ptz_keyframes.jpg').name}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Review PTZ behavior for a saved clip and sidecar log")
    parser.add_argument("--clip", required=True, type=Path, help="Path to the MP4 clip")
    parser.add_argument("--log-json", required=True, type=Path, help="Path to the clip .log.json sidecar")
    parser.add_argument("--journal", type=Path, help="Optional journalctl slice captured around the event")
    parser.add_argument("--out-dir", type=Path, help="Directory for report artifacts")
    args = parser.parse_args()

    output_dir = args.out_dir or args.log_json.parent / f"ptz_review_{args.log_json.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.clip))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open clip: {args.clip}")
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080.0
    capture.release()

    log = _load_json(args.log_json)
    centers = detection_rows(log, frame_width, frame_height)
    decisions = flatten_decisions(log)

    write_csv(
        output_dir / "ptz_centers.csv",
        centers,
        [
            "frame_idx", "event", "species", "confidence", "track_id",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "center_x_px", "center_y_px", "center_x_norm", "center_y_norm",
            "offset_x", "offset_y", "fill_pct",
        ],
    )
    write_csv(
        output_dir / "ptz_decisions.csv",
        decisions,
        [
            "timestamp", "time", "event", "mode", "source", "species", "track_id",
            "confidence", "bbox_px", "offset_x", "offset_y", "offset_magnitude",
            "pan", "tilt", "zoom", "pan_raw", "tilt_raw", "cap_active",
            "fill_pct", "frame_age_ms", "gap_since_last_move_ms",
            "gap_capture_to_capture_ms", "actual_slew_ms", "dispatch_late_ms",
        ],
    )
    render_contact_sheet(args.clip, centers, output_dir / "ptz_keyframes.jpg")
    report = build_report(
        log, centers, decisions, journal_perf_summary(args.journal), output_dir,
        stall_lines=journal_stall_summary(args.journal),
    )
    (output_dir / "ptz_review_report.md").write_text(report, encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
