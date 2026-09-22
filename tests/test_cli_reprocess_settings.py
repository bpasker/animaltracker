"""``reprocess`` analyses a clip the way the pipeline would (bug hunt item 1.26).

It built its detector from the legacy ``backend`` key, not
``postprocess_backend`` (they agree on production only by coincidence), and
its post-processor from the defaults with a ``--sample-rate`` of 5, not from
``general.clip``: the same clip came out differently from the command than
from the archive's Reanalyze. Now it uses ``create_postprocess_detector``
and ``build_processing_settings`` like every other caller, and
``--sample-rate`` is an override, not a default of its own.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from animaltracker import cli as cli_mod
from animaltracker import detector as detector_mod
from animaltracker.cli import build_parser, cmd_reprocess


def write_config(tmp_path) -> Path:
    cfg = tmp_path / "cameras.yml"
    cfg.write_text(textwrap.dedent(f"""
        general:
          storage_root: {tmp_path / 'storage'}
          logs_root: {tmp_path / 'logs'}
          notification:
            pushover_app_token_env: PUSHOVER_APP_TOKEN
            pushover_user_key_env: PUSHOVER_USER_KEY
          detector:
            backend: yolo
            postprocess_backend: speciesnet
            realtime_backend: megadetector
          clip:
            sample_rate: 2
            post_analysis_confidence: 0.35
        cameras: []
    """))
    (tmp_path / "storage" / "clips").mkdir(parents=True)
    return cfg


@pytest.fixture
def seen(monkeypatch):
    calls = {}

    def fake_postprocess_detector(detector_cfg):
        calls["postprocess_cfg"] = detector_cfg
        return SimpleNamespace(backend_name="speciesnet")

    def fake_create_detector(**kwargs):
        calls["create_detector"] = kwargs
        return SimpleNamespace(backend_name=kwargs["backend"])

    def fake_process_all(storage_root, detector, **kwargs):
        calls["process_all"] = dict(kwargs, detector=detector)
        return []

    monkeypatch.setattr(detector_mod, "create_postprocess_detector", fake_postprocess_detector)
    monkeypatch.setattr(detector_mod, "create_detector", fake_create_detector)
    import animaltracker.postprocess as pp
    monkeypatch.setattr(pp, "process_all_clips", fake_process_all)
    return calls


def args(cfg, **extra):
    base = dict(config=str(cfg), model=None, camera=None, clip=None, sample_rate=None,
                no_rename=False, no_thumbnails=False)
    base.update(extra)
    return SimpleNamespace(**base)


def test_it_uses_the_post_processing_backend_not_the_legacy_key(tmp_path, seen):
    cmd_reprocess(args(write_config(tmp_path)))

    assert seen["postprocess_cfg"].postprocess_backend == "speciesnet"
    assert "create_detector" not in seen, "the legacy path built a detector"
    assert seen["process_all"]["detector"].backend_name == "speciesnet"


def test_it_analyses_with_the_configured_settings(tmp_path, seen):
    cmd_reprocess(args(write_config(tmp_path)))

    settings = seen["process_all"]["settings"]
    assert settings.sample_rate == 2                     # from general.clip, not the old default 5
    assert settings.confidence_threshold == pytest.approx(0.35)


def test_sample_rate_on_the_command_line_is_an_override(tmp_path, seen):
    cmd_reprocess(args(write_config(tmp_path), sample_rate=7))

    assert seen["process_all"]["settings"].sample_rate == 7


def test_an_explicit_model_still_builds_that_model_on_the_post_processing_backend(tmp_path, seen):
    cmd_reprocess(args(write_config(tmp_path), model="models/big.pt"))

    assert seen["create_detector"]["backend"] == "speciesnet"
    assert seen["create_detector"]["model_path"] == "models/big.pt"


def test_the_parser_has_no_sample_rate_of_its_own():
    parsed = build_parser().parse_args(["--config", "x.yml", "reprocess"])
    assert parsed.sample_rate is None
