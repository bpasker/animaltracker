"""A second opinion on a finished clip from a local vision model.

After SpeciesNet has named a clip, four of its frames, each with the accepted
detection's box drawn on, go to a vision language model served by Ollama on
this machine, which answers one question: is there really a live animal in
the box? The answer, the animal it saw and a one-sentence description are
written into the clip's sidecar under ``review``.

It only observes. Nothing here deletes a clip, renames it or holds back an
alert; the verdicts are collected so they can be compared with what the clip
actually shows before anything is allowed to act on them. Measured on
2026-10-01 over 42 hand-labelled clips, ``qwen3.5:4b`` was right on all 38
it could be scored on (every Otteson2 ornament and lid false positive, every
real animal, including small night-vision ones), about 10 s a clip on the GTX
1080. Its animal names are poor at night (rabbits come back as cats), so the
species stays SpeciesNet's; only the yes/no is the model's.

The reviewer reads the clip and its sidecar back from disk, never the
post-processor's in-memory result, so the live path and the ``review``
command share one code path. Reviews run one at a time (``_REVIEW_LOCK``):
the model shares the GPU with the live detector, which slows from ~100 ms to
~250 ms a frame while one runs.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

FRAMES_PER_REVIEW = 4
FRAME_WIDTH = 1024
BOX_COLOUR = (0, 255, 255)  # yellow, as the prompt says

PROMPT = """These are {n} frames from one backyard security camera clip, in time order.
A wildlife detector drew a yellow box around something it thinks is an animal.
Decide whether there is really a live animal inside the yellow box in at least one frame.
Things that commonly fool the detector: garden ornaments, a metal lid or grill, chairs,
shadows, plants, and parts of people. A pet dog or cat counts as an animal.
The animal may be small, blurry, partly out of frame, or in black-and-white night vision.
Answer with JSON only."""
# Leave the prompt alone without re-running the 42-clip check: one added
# line ("Describe what happens in one short sentence of at most 20 words")
# turned 0 mistakes into 7, every one a small or night-vision animal called
# "none" or "shadow". Long descriptions are cut down in code instead
# (``first_sentence``).

SCHEMA = {
    "type": "object",
    "properties": {
        "real_animal": {"type": "boolean"},
        "animal": {"type": "string", "description": "common name, or 'none'"},
        "description": {"type": "string", "description": "one short sentence on what happens"},
    },
    "required": ["real_animal", "animal", "description"],
}

_REVIEW_LOCK = threading.Lock()


@dataclass(frozen=True)
class ReviewConfig:
    """The ``clip.review_*`` settings, read once per review so they apply live."""

    model: str
    url: str
    timeout_seconds: float

    @classmethod
    def from_clip_settings(cls, clip_cfg: Any) -> "ReviewConfig":
        return cls(
            model=str(getattr(clip_cfg, "review_model", "qwen3.5:4b") or "qwen3.5:4b"),
            url=str(getattr(clip_cfg, "review_url", "http://127.0.0.1:11434") or "http://127.0.0.1:11434").rstrip("/"),
            timeout_seconds=float(getattr(clip_cfg, "review_timeout_seconds", 120) or 120),
        )


def review_enabled(clip_cfg: Any) -> bool:
    return bool(getattr(clip_cfg, "review_enabled", False))


def sidecar_path(clip_path: Path) -> Path:
    return clip_path.with_suffix(".log.json")


def read_sidecar(clip_path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(sidecar_path(clip_path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def pick_frames(log_data: Dict[str, Any], n: int = FRAMES_PER_REVIEW) -> List[Tuple[int, List[float]]]:
    """``n`` accepted detections spread evenly over the clip, as (frame, box).

    Every accepted detection is a candidate, not only the main track's: the
    sidecar's track ids are renumbered when tracks merge, and the question is
    whether *anything* the clip was kept for is an animal.
    """
    entries = [
        e for e in (log_data.get("log_entries") or [])
        if isinstance(e, dict) and e.get("event") == "accepted"
        and isinstance(e.get("frame_idx"), int) and e["frame_idx"] >= 0
        and isinstance(e.get("bbox"), list) and len(e["bbox"]) == 4
    ]
    if not entries:
        return []
    entries.sort(key=lambda e: e["frame_idx"])
    if len(entries) <= n:
        chosen = entries
    else:
        chosen = [entries[round(i * (len(entries) - 1) / (n - 1))] for i in range(n)]
    return [(int(e["frame_idx"]), [float(v) for v in e["bbox"]]) for e in chosen]


def render_frames(clip_path: Path, picks: Sequence[Tuple[int, List[float]]]) -> List[str]:
    """The picked frames as base64 JPEGs, box drawn, at most FRAME_WIDTH wide."""
    import cv2

    images: List[str] = []
    cap = cv2.VideoCapture(str(clip_path))
    try:
        for frame_idx, bbox in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            if max(x1, y1, x2, y2) <= 1.0:  # normalised
                x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOUR, 3)
            if w > FRAME_WIDTH:
                frame = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                images.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    finally:
        cap.release()
    return images


def ask_model(cfg: ReviewConfig, images: Sequence[str]) -> Dict[str, Any]:
    """One chat call to Ollama; the parsed JSON answer."""
    body = {
        "model": cfg.model,
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [{"role": "user", "content": PROMPT.format(n=len(images)), "images": list(images)}],
    }
    req = urllib.request.Request(
        cfg.url + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as resp:
        reply = json.loads(resp.read())
    message = reply.get("message") if isinstance(reply, dict) else None
    answer = json.loads((message.get("content") if isinstance(message, dict) else None) or "{}")
    if not isinstance(answer, dict) or not isinstance(answer.get("real_animal"), bool):
        raise ValueError(f"unexpected answer from {cfg.model}: {answer!r}")
    return answer


def review_clip(clip_path: Path, cfg: ReviewConfig) -> Dict[str, Any]:
    """Review one clip. Always returns a record; a failure is recorded in ``error``."""
    record: Dict[str, Any] = {
        "model": cfg.model,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    picks = pick_frames(read_sidecar(clip_path))
    if not picks:
        record["error"] = "no accepted detections in the sidecar"
        return record
    record["frames"] = [f for f, _ in picks]
    started = time.monotonic()
    try:
        images = render_frames(clip_path, picks)
        if not images:
            raise ValueError("could not read any of the picked frames")
        with _REVIEW_LOCK:
            answer = ask_model(cfg, images)
    except Exception as exc:  # noqa: BLE001 - recorded, so a backfill goes on to the next clip
        record["error"] = str(getattr(exc, "reason", None) or exc) or type(exc).__name__
        record["seconds"] = round(time.monotonic() - started, 1)
        return record
    record["seconds"] = round(time.monotonic() - started, 1)
    record["real_animal"] = answer["real_animal"]
    record["animal"] = str(answer.get("animal") or "").strip()
    record["description"] = first_sentence(str(answer.get("description") or ""))
    return record


def first_sentence(text: str, limit: int = 200) -> str:
    """The model's description cut to its first sentence, at most ``limit`` characters."""
    text = " ".join(text.split())
    for i, ch in enumerate(text):
        if ch in ".!?" and (i + 1 == len(text) or text[i + 1] == " "):
            text = text[: i + 1]
            break
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def write_review(clip_path: Path, record: Dict[str, Any]) -> bool:
    """Merge ``record`` into the sidecar as ``review``, atomically.

    Only into a sidecar that already exists: a clip with none has not been
    analysed, and a sidecar holding nothing but a review would tell the
    recovery sweep the analysis is done.
    """
    path = sidecar_path(clip_path)
    data = read_sidecar(clip_path)
    if not data:
        return False
    data["review"] = record
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        mode = 0o644
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        # mkstemp makes the file 0600, and os.replace would carry that over
        # the sidecar every other reader of the archive can open.
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return True


def review_and_record(clip_path: Path, clip_cfg: Any, camera_id: str = "", species: str = "") -> Optional[Dict[str, Any]]:
    """Review a clip, store the verdict in its sidecar and log it. Never raises."""
    try:
        record = review_clip(clip_path, ReviewConfig.from_clip_settings(clip_cfg))
        write_review(clip_path, record)
    except Exception:  # noqa: BLE001 - a second opinion must never break an event
        LOGGER.exception("[REVIEW] %s: review of %s failed", camera_id, clip_path.name)
        return None
    if record.get("error"):
        LOGGER.warning("[REVIEW] %s: no verdict on %s: %s", camera_id, clip_path.name, record["error"])
    else:
        LOGGER.info(
            "[REVIEW] %s: %s %s (species %s, %.1fs): %s",
            camera_id, clip_path.name,
            "real animal" if record["real_animal"] else "NOT AN ANIMAL",
            species or "?", record.get("seconds", 0.0), record.get("description", ""),
        )
    return record
