"""The post-processor's sidecar names the thumbnails it wrote.

On the NFS archive a directory listing fetched between the clip rename and
the sidecar writes can stay cached and hide the thumbnails for good while
they open fine by name (prod, 2026-09-11). Readers find them by these names
instead of trusting a listing, and the writer drops the cached listing.
"""
import json

from animaltracker.postprocess import (
    ClipPostProcessor,
    ProcessingSettings,
    _drop_directory_cache,
)


def make_processor():
    proc = object.__new__(ClipPostProcessor)
    proc.settings = ProcessingSettings()
    proc.detector = object()
    return proc


def test_log_names_the_thumbnails_it_was_given(tmp_path):
    proc = make_processor()
    clip = tmp_path / "1789184388_mammalia_carnivora_canidae.mp4"
    thumb = tmp_path / "1789184388_mammalia_carnivora_canidae_thumb_mammalia_carnivora_canidae_t0.jpg"

    proc._save_processing_log(clip, [], None, {}, thumbnails=[thumb])

    log = json.loads(clip.with_suffix(".log.json").read_text())
    assert log["thumbnails"] == [{"file": thumb.name}]
    # Ahead of the long entries list, where someone reading the file sees it.
    assert list(log).index("thumbnails") < list(log).index("log_entries")


def test_log_omits_the_list_when_thumbnails_were_not_regenerated(tmp_path):
    proc = make_processor()
    clip = tmp_path / "1789184388_animal.mp4"

    proc._save_processing_log(clip, [], None, {})

    assert "thumbnails" not in json.loads(clip.with_suffix(".log.json").read_text())


def test_dropping_a_directory_cache_is_harmless(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")

    _drop_directory_cache(tmp_path)  # Linux: fadvise on the directory; macOS: no-op
    _drop_directory_cache(tmp_path / "missing")

    assert [p.name for p in tmp_path.iterdir()] == ["a.jpg"]
