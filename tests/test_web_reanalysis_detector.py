"""A reanalysis runs on the pipeline's detector, off the event loop, and
always gives the clip back.

Background (bug hunt, 2026-09-19): ``handle_reprocess`` called
``create_postprocess_detector`` directly in the coroutine, for every request.
That builds a whole SpeciesNet: for as long as the load took, every camera's
read loop, every MJPEG stream, every request and the PTZ dead-man waited
behind it, and each request put another model beside the pipeline's cached
one on a GPU with no room to spare. The clip was also claimed in the analysis
registry *before* that call and before the settings were parsed, outside the
``try``: a detector that would not load, or a ``settings`` that was not an
object, left the clip "Analyzing…" until the next restart, with every further
reanalysis answered 409 and the recovery sweep skipping it.
"""
from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from animaltracker import detector as detector_mod
from animaltracker import postprocess as pp
from animaltracker.analysis_recovery import ClipAnalysisRegistry
from animaltracker.web import WebServer

REL = "cam1/2026/09/10/1789000000_animal.mp4"


class JsonRequest:
    def __init__(self, body) -> None:
        self._body = body
        self.match_info = {}
        self.query = {}

    async def json(self):
        return self._body


class FakeWorker:
    """The slice of StreamWorker a reanalysis uses."""

    def __init__(self, name: str, fail: Exception | None = None) -> None:
        self.name = name
        self.fail = fail
        self.detector = SimpleNamespace(backend_name=f"{name}-realtime")
        self.threads: list = []

    def _get_postprocess_detector(self):
        self.threads.append(threading.current_thread())
        if self.fail is not None:
            raise self.fail
        return SimpleNamespace(backend_name=f"{self.name}-cached-speciesnet")


@pytest.fixture
def seen(monkeypatch):
    """Replace the post-processor with one that records what it was given."""
    record: dict = {"detectors": []}

    class FakeProcessor:
        def __init__(self, detector, storage_root, settings) -> None:
            record["detectors"].append(detector.backend_name)
            record["settings"] = settings

        def process_clip(self, path, update_filename, regenerate_thumbnails):
            if record.get("raise"):
                raise record["raise"]
            return SimpleNamespace(
                success=True, original_species="animal", new_species="animal", confidence=0.0,
                frames_analyzed=1, total_frames=2, raw_detections=0, filtered_detections=0,
                species_results={}, tracks_detected=0, thumbnails_saved=[], new_path=None,
                settings_used=None, error=None,
            )

    monkeypatch.setattr(pp, "ClipPostProcessor", FakeProcessor)

    def never(cfg):
        raise AssertionError("a reanalysis must not build a SpeciesNet of its own")

    monkeypatch.setattr(detector_mod, "create_postprocess_detector", never)
    return record


def make_server(tmp_path, workers):
    root = tmp_path / "storage"
    clip = root / "clips" / REL
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"a clip")
    registry = ClipAnalysisRegistry()
    server = WebServer(workers, root, tmp_path / "logs", port=0, analysis_registry=registry)
    server.runtime = SimpleNamespace(general=SimpleNamespace(
        detector=SimpleNamespace(), clip=SimpleNamespace(sample_rate=2)))
    return server, registry, clip


def reanalyse(server, body):
    loop_thread: list = []

    async def run():
        loop_thread.append(threading.current_thread())
        return await server.handle_reprocess(JsonRequest(body))

    resp = asyncio.run(run())
    payload = json.loads(resp.body.decode()) if resp.body and resp.body[:1] == b"{" else None
    return resp, payload, loop_thread[0]


def assert_released(server, registry, clip) -> None:
    assert not registry.is_active(clip)
    assert server.reprocessing_jobs == {}


# --- whose detector, and on which thread ----------------------------------------

def test_it_runs_on_the_cached_detector_of_the_clip_s_own_camera(tmp_path, seen):
    workers = {"cam2": FakeWorker("cam2"), "cam1": FakeWorker("cam1")}
    server, registry, clip = make_server(tmp_path, workers)

    resp, payload, loop_thread = reanalyse(server, {"path": REL})

    assert resp.status == 200 and payload["success"] is True
    assert seen["detectors"] == ["cam1-cached-speciesnet"]
    # Fetched on a worker thread: the first use after a restart loads the model.
    assert workers["cam1"].threads and workers["cam1"].threads[0] is not loop_thread
    assert workers["cam2"].threads == []
    assert_released(server, registry, clip)


def test_a_retired_camera_s_clip_borrows_any_worker_s_detector(tmp_path, seen):
    server, registry, clip = make_server(tmp_path, {"cam9": FakeWorker("cam9")})

    resp, payload, _ = reanalyse(server, {"path": REL})

    assert resp.status == 200
    assert seen["detectors"] == ["cam9-cached-speciesnet"]


def test_a_server_with_no_pipeline_builds_one_from_the_configuration(tmp_path, seen, monkeypatch):
    built: list = []

    def build(cfg):
        built.append(threading.current_thread())
        return SimpleNamespace(backend_name="built-from-config")

    monkeypatch.setattr(detector_mod, "create_postprocess_detector", build)
    server, registry, clip = make_server(tmp_path, {})

    resp, payload, loop_thread = reanalyse(server, {"path": REL})

    assert resp.status == 200 and seen["detectors"] == ["built-from-config"]
    assert built and built[0] is not loop_thread        # even then, not on the event loop


# --- the claim is always given back --------------------------------------------------

def test_a_detector_that_will_not_load_answers_500_and_releases_the_clip(tmp_path, seen):
    workers = {"cam1": FakeWorker("cam1", fail=RuntimeError("CUDA out of memory"))}
    server, registry, clip = make_server(tmp_path, workers)

    resp, payload, _ = reanalyse(server, {"path": REL})

    assert resp.status == 500
    assert payload == {"success": False, "error": "CUDA out of memory"}
    assert_released(server, registry, clip)

    # ...so the next attempt is a real attempt, not a 409, and the sweep can have it.
    workers["cam1"].fail = None
    resp, payload, _ = reanalyse(server, {"path": REL})
    assert resp.status == 200


def test_a_post_processor_that_raises_releases_the_clip(tmp_path, seen):
    server, registry, clip = make_server(tmp_path, {"cam1": FakeWorker("cam1")})
    seen["raise"] = ValueError("bad frame")

    resp, payload, _ = reanalyse(server, {"path": REL})

    assert resp.status == 500 and payload["error"] == "bad frame"
    assert_released(server, registry, clip)


@pytest.mark.parametrize("settings", [["sample_rate", 5], "fast", 3])
def test_settings_that_are_not_an_object_are_a_bad_request_and_claim_nothing(tmp_path, seen, settings):
    worker = FakeWorker("cam1")
    server, registry, clip = make_server(tmp_path, {"cam1": worker})

    resp, _, _ = reanalyse(server, {"path": REL, "settings": settings})

    assert resp.status == 400
    assert worker.threads == [] and seen["detectors"] == []
    assert_released(server, registry, clip)


def test_no_settings_at_all_is_fine_and_overrides_still_win(tmp_path, seen):
    server, registry, clip = make_server(tmp_path, {"cam1": FakeWorker("cam1")})

    assert reanalyse(server, {"path": REL, "settings": None})[0].status == 200
    assert seen["settings"].sample_rate == 2             # from the configuration

    assert reanalyse(server, {"path": REL, "settings": {"sample_rate": 5}})[0].status == 200
    assert seen["settings"].sample_rate == 5


def test_the_clip_is_held_while_the_job_runs(tmp_path, seen, monkeypatch):
    server, registry, clip = make_server(tmp_path, {"cam1": FakeWorker("cam1")})
    held: list = []
    original = pp.ClipPostProcessor.process_clip

    def process_clip(self, path, update_filename, regenerate_thumbnails):
        held.append((registry.is_active(clip), REL in server.reprocessing_jobs))
        return original(self, path, update_filename, regenerate_thumbnails)

    monkeypatch.setattr(pp.ClipPostProcessor, "process_clip", process_clip)

    assert reanalyse(server, {"path": REL})[0].status == 200
    assert held == [(True, True)]
    assert_released(server, registry, clip)
