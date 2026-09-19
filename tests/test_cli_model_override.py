"""``run`` takes its YOLO model from the configuration unless told otherwise.

Background (bug hunt, 2026-09-19): ``run --model`` defaulted to
``"yolov8n.pt"`` and ``PipelineOrchestrator`` applied any truthy value over
``detector.model_path``. The service starts ``run`` with no ``--model``, so
a model named in ``cameras.yml`` or on the settings page never took effect:
``yolov8n.pt`` was loaded from the working directory instead. The override
also rewrote the runtime config, and ``detector.model_path`` is one of the
fields the pending-restart banner diffs against the file, so the banner said
"the YOLO model path changed" for ever, restart or not.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from animaltracker import pipeline as pipeline_mod
from animaltracker.cli import build_parser
from animaltracker.config import load_runtime_config
from animaltracker.pipeline import PipelineOrchestrator

SAMPLE = Path(__file__).resolve().parents[1] / "config" / "cameras.sample.yml"


def test_run_has_no_model_of_its_own():
    assert build_parser().parse_args(["--config", "x.yml", "run"]).model is None
    assert build_parser().parse_args(["--config", "x.yml", "run", "--model", "big.pt"]).model == "big.pt"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    built: list = []

    def fake_detector(detector_cfg):
        built.append(detector_cfg.model_path)
        return SimpleNamespace(backend_name="yolo")

    monkeypatch.setattr(pipeline_mod, "create_realtime_detector", fake_detector)
    cfg = load_runtime_config(SAMPLE)
    cfg.general.storage_root = str(tmp_path / "storage")
    cfg.general.logs_root = str(tmp_path / "logs")
    cfg.general.detector.model_path = "models/yolo11m.pt"
    return cfg, built


def test_the_configured_model_is_the_one_that_loads(runtime):
    cfg, built = runtime

    PipelineOrchestrator(runtime=cfg)

    assert built == ["models/yolo11m.pt"]
    assert cfg.general.detector.model_path == "models/yolo11m.pt"      # file and runtime still agree


def test_an_explicit_model_still_wins(runtime):
    cfg, built = runtime

    PipelineOrchestrator(runtime=cfg, model_path="models/custom.pt")

    assert built == ["models/custom.pt"]
