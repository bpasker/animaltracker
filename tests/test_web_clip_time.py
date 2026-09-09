"""clip_start_time() and primary_thumbnail() — the two card-level helpers."""
from pathlib import Path
from types import SimpleNamespace

import animaltracker.web as web_mod


def _stat(mtime):
    return SimpleNamespace(st_mtime=mtime, st_size=10)


def test_epoch_prefix_is_the_clip_time():
    # Real case: detection began 08:12:01, the transcode finished 08:12:38.
    t = web_mod.clip_start_time(
        Path("1788873121_mammalia_rodentia_myocastoridae.mp4"), _stat(1788873158.36))
    assert t.timestamp() == 1788873121
    assert t.tzinfo is web_mod.CENTRAL_TZ


def test_no_epoch_prefix_falls_back_to_mtime():
    t = web_mod.clip_start_time(Path("imported_deer.mp4"), _stat(1700000000.0))
    assert t.timestamp() == 1700000000.0


def test_short_numeric_prefix_is_not_an_epoch():
    t = web_mod.clip_start_time(Path("42_deer.mp4"), _stat(1700000000.0))
    assert t.timestamp() == 1700000000.0


def test_epoch_later_than_the_file_falls_back_to_mtime():
    t = web_mod.clip_start_time(Path("4102444800_deer.mp4"), _stat(1700000000.0))
    assert t.timestamp() == 1700000000.0


def test_epoch_slightly_after_mtime_is_tolerated():
    # Clock skew between the camera host and storage: up to a day is accepted.
    t = web_mod.clip_start_time(Path("1700000100_deer.mp4"), _stat(1700000000.0))
    assert t.timestamp() == 1700000100


def test_primary_thumbnail_prefers_the_clips_own_species():
    clip = {
        "species": "Dog/Canid",
        "thumbnails": [
            {"url": "/clips/a_thumb_mammalia_carnivora_carnivorous_mammal_t0.jpg",
             "species": "Carnivore (Cat/Dog/Raccoon)"},
            {"url": "/clips/a_thumb_mammalia_carnivora_canidae_t1.jpg", "species": "Dog/Canid"},
        ],
    }
    assert web_mod.primary_thumbnail(clip).endswith("canidae_t1.jpg")


def test_primary_thumbnail_falls_back_to_the_first_track():
    clip = {"species": "Deer", "thumbnails": [
        {"url": "/clips/x_t0.jpg", "species": "Animal"},
        {"url": "/clips/x_t1.jpg", "species": "Mammal"},
    ]}
    assert web_mod.primary_thumbnail(clip) == "/clips/x_t0.jpg"


def test_primary_thumbnail_is_none_without_thumbnails():
    assert web_mod.primary_thumbnail({"species": "Deer", "thumbnails": []}) is None
    assert web_mod.primary_thumbnail({"species": "Deer"}) is None
