"""One mapping from ``general.clip`` to ``ProcessingSettings`` for every
caller — the live event, a reanalysis and the recovery sweep — so a clip
analysed later comes out the same as it would have at the time."""
from types import SimpleNamespace

from animaltracker.postprocess import ProcessingSettings, build_processing_settings


def clip_cfg(**overrides):
    values = dict(
        sample_rate=2, post_analysis_confidence=0.3, post_analysis_generic_confidence=0.8,
        tracking_enabled=True, track_merge_gap=60, spatial_merge_enabled=True,
        spatial_merge_iou=0.6, spatial_merge_reach=1.5, hierarchical_merge_enabled=False,
        single_animal_mode=True, thumbnail_cropped=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_config_values_map_onto_the_settings():
    s = build_processing_settings(clip_cfg())
    assert s.sample_rate == 2
    assert s.confidence_threshold == 0.3
    assert s.generic_confidence == 0.8
    assert s.tracking_enabled is True
    assert s.same_species_merge_gap == 60
    assert s.hierarchical_merge_gap == 60      # one gap setting, both merges
    assert s.spatial_merge_enabled is True
    assert s.spatial_merge_iou == 0.6
    assert s.spatial_merge_reach == 1.5
    assert s.spatial_merge_gap == 30
    assert s.hierarchical_merge_enabled is False
    assert s.single_animal_mode is True
    assert s.thumbnail_cropped is False
    assert s.merge_enabled is True
    assert s.save_processing_log is True


def test_missing_attributes_and_no_config_fall_back_to_the_defaults():
    defaults = ProcessingSettings()
    s = build_processing_settings(None)
    assert s.sample_rate == defaults.sample_rate
    assert s.confidence_threshold == defaults.confidence_threshold
    assert s.same_species_merge_gap == 120
    assert s.hierarchical_merge_gap == 120
    assert s.thumbnail_cropped is True

    sparse = build_processing_settings(SimpleNamespace(sample_rate=7))
    assert sparse.sample_rate == 7
    assert sparse.spatial_merge_iou == 0.3


def test_overrides_from_a_reanalysis_request_win():
    s = build_processing_settings(clip_cfg(), {"sample_rate": 9, "hierarchical_merge_gap": 5,
                                               "lost_track_buffer": 40})
    assert s.sample_rate == 9
    assert s.hierarchical_merge_gap == 5
    assert s.same_species_merge_gap == 60       # untouched by the override
    assert s.lost_track_buffer == 40
