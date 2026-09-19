"""What the JSON API says about a clip that has no key frames yet.

``analysis`` is ``running`` while a job holds the clip, ``queued`` when its
post-processing never finished and the recovery sweeper will get to it,
``unfinished`` when nothing will, and absent for every analysed clip. The
archive turns that into "Analyzing…", "Awaiting analysis" or "Not analyzed"
instead of a bare "No frame".
"""
import asyncio
import json
import os
import time
from types import SimpleNamespace

import pytest

from animaltracker.analysis_recovery import ClipAnalysisRegistry
from animaltracker.web import WebServer


class FakeRequest:
    def __init__(self, query=None, match_info=None, query_string=""):
        self.query = query or {}
        self.match_info = match_info or {}
        self.query_string = query_string


class FakeRecovery:
    def __init__(self, enabled=True, running=True):
        self.enabled = enabled
        self.running = running

    def status(self):
        return {"enabled": self.enabled, "running": self.running}


def call(handler, **kw):
    resp = asyncio.run(handler(FakeRequest(**kw)))
    return resp.status, json.loads(resp.body.decode())


def make_clip(clips_dir, camera, epoch, label, *, sidecar=False, thumb=False):
    when = time.localtime(epoch)
    d = clips_dir / camera / f"{when.tm_year:04d}" / f"{when.tm_mon:02d}" / f"{when.tm_mday:02d}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{int(epoch)}_{label}.mp4"
    f.write_bytes(b"x" * 1024)
    os.utime(f, (epoch, epoch))
    if sidecar:
        f.with_name(f"{int(epoch)}_{label}.log.json").write_text("{}")
    if thumb:
        f.with_name(f"{int(epoch)}_{label}_thumb_{label}_t0.jpg").write_bytes(b"\xff\xd8\xff")
    return f


@pytest.fixture
def base():
    return time.mktime((2026, 9, 14, 6, 0, 0, 0, 0, -1))


def make_server(tmp_path, **kw):
    return WebServer({}, tmp_path, tmp_path / "logs", port=0, **kw)


# ---------------------------------------------------------------- the scan

def test_scan_marks_a_clip_whose_analysis_never_finished(tmp_path, base):
    server = make_server(tmp_path)
    clips = tmp_path / "clips"
    make_clip(clips, "cam1", base, "animal")
    make_clip(clips, "cam1", base + 10, "animal", sidecar=True)
    make_clip(clips, "cam1", base + 20, "mammalia_carnivora_canidae", thumb=True)

    by_name = {c["filename"]: c for c in server._scan_recordings()}
    assert by_name[f"{int(base)}_animal.mp4"]["unfinished"] is True
    assert by_name[f"{int(base + 10)}_animal.mp4"]["unfinished"] is False
    assert by_name[f"{int(base + 20)}_mammalia_carnivora_canidae.mp4"]["unfinished"] is False


def test_a_sidecar_with_only_ptz_decisions_is_still_unfinished(tmp_path, base):
    """What a failed analysis leaves on a PTZ camera: the pipeline parked the
    event's PTZ decisions in the sidecar and there was no analysis to add them
    to. It used to read as analysed, a "No frame" card nothing would fix."""
    server = make_server(tmp_path, recovery=FakeRecovery())
    clips = tmp_path / "clips"
    clip = make_clip(clips, "cam2", base, "animal")
    clip.with_name(f"{int(base)}_animal.log.json").write_text(json.dumps(
        {"clip": clip.name, "ptz_decisions": [{"timestamp": base + 1, "event": "move"}]}, indent=2))

    (scanned,) = server._scan_recordings()
    assert scanned["unfinished"] is True

    rel = str(clip.relative_to(clips))
    status, detail = call(server.handle_clip_api, match_info={"path": rel})
    assert status == 200 and detail["analysis"] == "queued"


def test_manual_clips_are_never_unfinished(tmp_path):
    server = make_server(tmp_path)
    clips = tmp_path / "clips"
    clips.mkdir(exist_ok=True)
    (clips / "manual_cam1_1789384540.mp4").write_bytes(b"x")

    (clip,) = server._scan_recordings()
    assert clip["species"] == "Manual clip"
    assert clip["unfinished"] is False


# ------------------------------------------------------------ list payload

def test_list_reports_unfinished_when_nothing_will_pick_the_clip_up(tmp_path, base):
    server = make_server(tmp_path)
    make_clip(tmp_path / "clips", "cam1", base, "animal")
    make_clip(tmp_path / "clips", "cam1", base + 20, "mammalia_carnivora_canidae", thumb=True)

    status, body = call(server.handle_recordings_api, query={"sort": "oldest"})
    assert status == 200
    unfinished, classified = body["clips"]
    assert unfinished["analysis"] == "unfinished"
    assert unfinished["thumbnail"] is None
    assert "analysis" not in classified


def test_list_reports_queued_while_the_sweeper_is_on(tmp_path, base):
    server = make_server(tmp_path, analysis_registry=ClipAnalysisRegistry(), recovery=FakeRecovery())
    make_clip(tmp_path / "clips", "cam1", base, "animal")

    status, body = call(server.handle_recordings_api)
    assert body["clips"][0]["analysis"] == "queued"


def test_list_reports_unfinished_when_recovery_is_switched_off(tmp_path, base):
    server = make_server(tmp_path, analysis_registry=ClipAnalysisRegistry(),
                         recovery=FakeRecovery(enabled=False))
    make_clip(tmp_path / "clips", "cam1", base, "animal")

    status, body = call(server.handle_recordings_api)
    assert body["clips"][0]["analysis"] == "unfinished"


def test_list_reports_running_while_a_job_holds_the_clip(tmp_path, base):
    registry = ClipAnalysisRegistry()
    server = make_server(tmp_path, analysis_registry=registry, recovery=FakeRecovery())
    clip = make_clip(tmp_path / "clips", "cam1", base, "animal")
    registry.begin(clip, "recovery")

    status, body = call(server.handle_recordings_api)
    assert body["clips"][0]["analysis"] == "running"

    registry.end(clip)
    server._scan_cache = None   # the registry is read per request; the scan is cached
    status, body = call(server.handle_recordings_api)
    assert body["clips"][0]["analysis"] == "queued"


def test_running_is_reported_for_a_classified_clip_being_reanalysed(tmp_path, base):
    """A reanalysis of a finished clip shows as running too: the card should
    say so rather than flash its old key frame as if nothing were happening."""
    registry = ClipAnalysisRegistry()
    server = make_server(tmp_path, analysis_registry=registry)
    clip = make_clip(tmp_path / "clips", "cam1", base, "mammalia_carnivora_canidae", sidecar=True, thumb=True)
    registry.begin(clip, "reanalyze")

    status, body = call(server.handle_recordings_api)
    assert body["clips"][0]["analysis"] == "running"
    assert body["clips"][0]["thumbnail"]   # the frame is still there to show


# ------------------------------------------------------------- day payload

def test_day_payload_carries_the_state(tmp_path, base):
    server = make_server(tmp_path, analysis_registry=ClipAnalysisRegistry(), recovery=FakeRecovery())
    make_clip(tmp_path / "clips", "cam1", base, "animal")
    make_clip(tmp_path / "clips", "cam1", base + 20, "mammalia_carnivora_canidae", thumb=True)

    date = time.strftime("%Y-%m-%d", time.localtime(base))
    status, body = call(server.handle_day_api, match_info={"date": date})
    assert status == 200
    states = {c["filename"]: c.get("analysis") for c in body["clips"]}
    assert states[f"{int(base)}_animal.mp4"] == "queued"
    assert states[f"{int(base + 20)}_mammalia_carnivora_canidae.mp4"] is None


# ---------------------------------------------------------- detail payload

def test_detail_reports_the_state_and_reprocessing_covers_pipeline_jobs(tmp_path, base):
    registry = ClipAnalysisRegistry()
    server = make_server(tmp_path, analysis_registry=registry, recovery=FakeRecovery())
    clip = make_clip(tmp_path / "clips", "cam1", base, "animal")
    rel = str(clip.relative_to(tmp_path / "clips"))

    status, body = call(server.handle_clip_api, match_info={"path": rel})
    assert status == 200
    assert body["analysis"] == "queued"
    assert body["reprocessing"] is False
    assert "unfinished" not in body

    registry.begin(clip, "event")
    status, body = call(server.handle_clip_api, match_info={"path": rel})
    assert body["analysis"] == "running"
    assert body["reprocessing"] is True


def test_detail_of_a_renamed_clip_says_where_it_went(tmp_path, base):
    server = make_server(tmp_path)
    clips = tmp_path / "clips"
    renamed = make_clip(clips, "cam1", base, "mammalia_carnivora_canidae", sidecar=True)
    old_rel = str(renamed.relative_to(clips)).replace("mammalia_carnivora_canidae", "animal")

    status, body = call(server.handle_clip_api, match_info={"path": old_rel})
    assert status == 404
    assert body["renamed_to"] == str(renamed.relative_to(clips))


def test_detail_of_a_missing_clip_without_a_successor_is_a_plain_404(tmp_path, base):
    server = make_server(tmp_path)
    clips = tmp_path / "clips"
    make_clip(clips, "cam1", base, "mammalia_carnivora_canidae")
    make_clip(clips, "cam1", base, "bird")   # two candidates: ambiguous, so none
    old_rel = f"cam1/{time.strftime('%Y/%m/%d', time.localtime(base))}/{int(base)}_animal.mp4"

    status, body = call(server.handle_clip_api, match_info={"path": old_rel})
    assert status == 404
    assert "renamed_to" not in body

    status, body = call(server.handle_clip_api, match_info={"path": "cam1/2026/01/01/1_deer.mp4"})
    assert status == 404
    assert "renamed_to" not in body


# ------------------------------------------------------- reanalysis handler

class JsonRequest(FakeRequest):
    def __init__(self, body):
        super().__init__()
        self._body = body

    async def json(self):
        return self._body


def test_reanalysis_refuses_a_clip_the_pipeline_holds(tmp_path, base):
    registry = ClipAnalysisRegistry()
    server = make_server(tmp_path, analysis_registry=registry)
    clip = make_clip(tmp_path / "clips", "cam1", base, "animal")
    rel = str(clip.relative_to(tmp_path / "clips"))
    registry.begin(clip, "recovery")

    resp = asyncio.run(server.handle_reprocess(JsonRequest({"path": rel})))
    assert resp.status == 409
    assert "already analysing" in json.loads(resp.body.decode())["error"]
    assert rel not in server.reprocessing_jobs


def test_reanalysis_claims_the_clip_and_releases_it_afterwards(tmp_path, base, monkeypatch):
    registry = ClipAnalysisRegistry()
    server = make_server(tmp_path, analysis_registry=registry)
    server.runtime = SimpleNamespace(general=SimpleNamespace(
        detector=SimpleNamespace(), clip=SimpleNamespace(sample_rate=2)))
    clip = make_clip(tmp_path / "clips", "cam1", base, "animal")
    rel = str(clip.relative_to(tmp_path / "clips"))

    from animaltracker import detector as detector_mod, postprocess as pp
    monkeypatch.setattr(detector_mod, "create_postprocess_detector",
                        lambda cfg: SimpleNamespace(backend_name="fake"))
    seen = {}

    class FakeProcessor:
        def __init__(self, detector, storage_root, settings):
            seen["settings"] = settings
            seen["held_during_job"] = registry.is_active(clip)

        def process_clip(self, path, update_filename, regenerate_thumbnails):
            return SimpleNamespace(
                success=True, original_species="animal", new_species="animal", confidence=0.0,
                frames_analyzed=1, total_frames=2, raw_detections=0, filtered_detections=0,
                species_results={}, tracks_detected=0, thumbnails_saved=[], new_path=None,
                settings_used=None, error=None,
            )

    monkeypatch.setattr(pp, "ClipPostProcessor", FakeProcessor)

    resp = asyncio.run(server.handle_reprocess(JsonRequest({"path": rel, "settings": {"sample_rate": 5}})))
    assert resp.status == 200, resp.body
    assert seen["held_during_job"] is True
    assert seen["settings"].sample_rate == 5          # the request's override wins
    assert not registry.is_active(clip)
    assert rel not in server.reprocessing_jobs
