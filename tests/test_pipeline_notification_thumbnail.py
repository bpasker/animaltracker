"""pick_notification_thumbnail chooses the alert photo from the files the
post-processor just wrote, checked by name, because a directory listing
taken right after the clip rename can be stale on the NFS archive."""
from pathlib import Path

from animaltracker.pipeline import pick_notification_thumbnail

STEM = "1789184388_mammalia_carnivora_canidae"


def make_clip(tmp_path):
    clip = tmp_path / f"{STEM}.mp4"
    clip.write_bytes(b"\x00")
    return clip


def thumb(clip, species, idx):
    path = clip.with_name(f"{STEM}_thumb_{species}_t{idx}.jpg")
    path.write_bytes(b"\xff\xd8\xff")
    return path


def test_saved_thumbnail_wins_even_when_the_listing_hides_it(tmp_path, monkeypatch):
    clip = make_clip(tmp_path)
    saved = thumb(clip, "mammalia_carnivora_canidae", 0)
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter(()))

    assert pick_notification_thumbnail([saved], clip, "mammalia_carnivora_canidae") == saved


def test_the_notified_species_track_is_preferred_over_an_earlier_generic_one(tmp_path):
    clip = make_clip(tmp_path)
    generic = thumb(clip, "animal", 0)
    dog = thumb(clip, "mammalia_carnivora_canidae", 1)

    assert pick_notification_thumbnail([generic, dog], clip, "mammalia_carnivora_canidae") == dog
    assert pick_notification_thumbnail([generic, dog], clip, "bird") == generic


def test_species_with_spaces_matches_the_writer_naming(tmp_path):
    clip = make_clip(tmp_path)
    generic = thumb(clip, "animal", 0)
    carnivore = thumb(clip, "mammalia_carnivora_carnivorous_mammal", 1)

    chosen = pick_notification_thumbnail([generic, carnivore], clip, "mammalia_carnivora_carnivorous mammal")

    assert chosen == carnivore


def test_saved_paths_that_are_gone_fall_back_to_the_listing(tmp_path):
    clip = make_clip(tmp_path)
    on_disk = thumb(clip, "animal", 0)
    gone = clip.with_name(f"{STEM}_thumb_animal_t1.jpg")

    assert pick_notification_thumbnail([gone], clip, "animal") == on_disk


def test_no_thumbnail_anywhere_gives_none(tmp_path):
    clip = make_clip(tmp_path)

    assert pick_notification_thumbnail([], clip, "animal") is None
