"""The local vision model's second opinion on a clip (review.py).

It observes only: it reads the clip and its sidecar back from disk, asks
Ollama about four boxed frames and stores the answer in the sidecar under
``review``. A failure of any kind is recorded, never raised into the event.
"""
from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from animaltracker import review
from animaltracker.config import ClipSettings

W, H, FRAMES = 320, 180, 40


def make_clip(tmp_path: Path, accepted_frames=(2, 10, 20, 30, 38), extra=None) -> Path:
    clip = tmp_path / "1790000000_mammalia_carnivora_canidae.mp4"
    out = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    for i in range(FRAMES):
        frame = np.full((H, W, 3), 40, np.uint8)
        cv2.putText(frame, str(i), (10, 40), 0, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    entries = [{"frame_idx": f, "event": "accepted", "species": "dog", "confidence": 0.9,
                "track_id": None, "bbox": [50.0, 50.0, 120.0, 110.0]} for f in accepted_frames]
    entries += [{"frame_idx": 5, "event": "detector_filtered", "species": "blank",
                 "confidence": 0.0, "track_id": None, "bbox": [0, 0, 1, 1]}]
    sidecar = {"clip": clip.name, "tracking_summary": {"tracks": []}, "log_entries": entries}
    sidecar.update(extra or {})
    clip.with_suffix(".log.json").write_text(json.dumps(sidecar))
    return clip


class FakeOllama:
    """A one-endpoint stand-in for Ollama's /api/chat."""

    def __init__(self, answer):
        self.answer = answer
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                reply = {"message": {"role": "assistant", "content": json.dumps(outer.answer)}}
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake_ollama():
    servers = []

    def start(answer):
        s = FakeOllama(answer)
        servers.append(s)
        return s

    yield start
    for s in servers:
        s.close()


def clip_cfg(url: str, **kw) -> SimpleNamespace:
    return SimpleNamespace(review_enabled=True, review_model="qwen3.5:4b", review_url=url,
                           review_timeout_seconds=10, **kw)


def test_off_by_default():
    assert ClipSettings().review_enabled is False
    assert review.review_enabled(ClipSettings()) is False


def test_four_accepted_detections_spread_over_the_clip():
    log = {"log_entries": [
        {"frame_idx": f, "event": "accepted", "bbox": [1, 2, 3, 4]} for f in (30, 0, 10, 20, 40, 50, 60)
    ] + [{"frame_idx": 5, "event": "tracked", "bbox": [1, 2, 3, 4]},
         {"frame_idx": 7, "event": "accepted", "bbox": None}]}
    assert [f for f, _ in review.pick_frames(log)] == [0, 20, 40, 60]


def test_fewer_detections_than_frames_takes_them_all_and_none_takes_none():
    log = {"log_entries": [{"frame_idx": 3, "event": "accepted", "bbox": [1, 2, 3, 4]}]}
    assert review.pick_frames(log) == [(3, [1.0, 2.0, 3.0, 4.0])]
    assert review.pick_frames({}) == []


def test_a_verdict_is_asked_for_and_stored_beside_the_analysis(tmp_path, fake_ollama):
    server = fake_ollama({"real_animal": False, "animal": "none",
                          "description": "A bird-shaped garden ornament that never moves."})
    clip = make_clip(tmp_path, extra={"ptz_decisions": [{"t": 1}]})

    record = review.review_and_record(clip, clip_cfg(server.url), "Otteson2", "bird")

    assert record["real_animal"] is False
    path, body = server.requests[0]
    assert path == "/api/chat"
    assert body["model"] == "qwen3.5:4b" and body["format"] == review.SCHEMA
    images = body["messages"][0]["images"]
    assert len(images) == 4
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(images[0]), np.uint8), cv2.IMREAD_COLOR)
    assert frame.shape[:2] == (H, W)
    assert tuple(int(c) for c in frame[50, 85]) != (40, 40, 40)  # the box is drawn

    stored = json.loads(clip.with_suffix(".log.json").read_text())
    assert stored["review"]["real_animal"] is False
    assert stored["review"]["frames"] == [2, 10, 30, 38]
    # Everything the post-processor and the pipeline wrote is still there.
    assert stored["ptz_decisions"] == [{"t": 1}]
    assert len(stored["log_entries"]) == 6
    assert not list(tmp_path.glob("*.tmp"))


def test_no_server_records_the_error_and_keeps_the_clip(tmp_path):
    clip = make_clip(tmp_path)
    record = review.review_and_record(clip, clip_cfg("http://127.0.0.1:9"), "cam1", "dog")
    assert "error" in record and "real_animal" not in record
    assert clip.exists()
    assert "error" in json.loads(clip.with_suffix(".log.json").read_text())["review"]


def test_an_answer_without_a_verdict_is_an_error(tmp_path, fake_ollama):
    server = fake_ollama({"animal": "dog"})
    clip = make_clip(tmp_path)
    record = review.review_clip(clip, review.ReviewConfig.from_clip_settings(clip_cfg(server.url)))
    assert "error" in record and "real_animal" not in record


def test_a_clip_with_no_detections_is_not_sent(tmp_path, fake_ollama):
    server = fake_ollama({"real_animal": True, "animal": "dog", "description": "x"})
    clip = make_clip(tmp_path, accepted_frames=())
    record = review.review_clip(clip, review.ReviewConfig.from_clip_settings(clip_cfg(server.url)))
    assert record["error"] == "no accepted detections in the sidecar"
    assert server.requests == []


def test_no_sidecar_is_never_created(tmp_path):
    clip = tmp_path / "1790000000_animal.mp4"
    clip.write_bytes(b"")
    assert review.write_review(clip, {"real_animal": True}) is False
    assert not clip.with_suffix(".log.json").exists()


def test_a_crash_inside_the_review_never_reaches_the_event(tmp_path, monkeypatch):
    clip = make_clip(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("cv2 exploded")

    monkeypatch.setattr(review, "review_clip", boom)
    assert review.review_and_record(clip, clip_cfg("http://127.0.0.1:9")) is None


def test_the_pipeline_reviews_only_after_an_alert_and_only_when_enabled():
    import inspect
    from animaltracker import pipeline

    src = inspect.getsource(pipeline.StreamWorker)
    assert "return clip_path, final_species" in src
    assert "if alerted and review_enabled(clip_cfg):" in src
    # The review runs before the clip's claim is released.
    body = src[src.index("def analyse() -> None:"):]
    assert body.index("review_and_record(") < body.index("release()")


def test_a_long_description_is_cut_to_its_first_sentence():
    assert review.first_sentence("A dog walks by. It stops at the bowl.") == "A dog walks by."
    assert review.first_sentence("  A 3.5 kg  cat\nsits ") == "A 3.5 kg cat sits"
    long = "word " * 100
    out = review.first_sentence(long)
    assert len(out) == 200 and out.endswith("…")


def test_the_prompt_is_the_one_that_was_measured():
    """Change the prompt only after re-running the 42-clip check on prod
    (/tmp/vlm-evalvlm.py): one extra line about keeping the description short
    took it from 0 mistakes to 7, all of them small or night-time animals
    called "none" or "shadow". Then update this hash."""
    import hashlib

    assert hashlib.sha256(review.PROMPT.encode()).hexdigest()[:16] == "680f87caff754090"
