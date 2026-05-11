from animaltracker.detector import SpeciesNetDetector
from animaltracker.postprocess import ClipPostProcessor
from animaltracker.tracker import ObjectTracker, TrackInfo


def test_speciesnet_collapses_generic_reptile_taxonomy() -> None:
    taxonomy = "1F689929-1234-4567-89ab-0123456789ab;Reptilia;;;;;Reptile"

    assert SpeciesNetDetector._simplify_species_name(None, taxonomy) == "reptile"


def test_legacy_reptilia_reptile_is_class_level() -> None:
    processor = object.__new__(ClipPostProcessor)
    track = TrackInfo(track_id=1)
    tracker = object.__new__(ObjectTracker)

    assert processor._calculate_specificity("reptilia_reptile") == 1
    assert track._calculate_specificity("reptilia_reptile") == 1
    assert tracker._get_species_hierarchy("reptilia_reptile") == ("reptile", 1)


def test_edge_anchored_generic_reptile_pipe_is_filtered() -> None:
    pipe_bbox = [1.0, 172.0, 889.0, 359.0]

    assert SpeciesNetDetector._is_edge_anchored_generic_reptile(
        "reptile", pipe_bbox, (1080, 1920, 3)
    )


def test_centered_generic_reptile_is_not_pipe_filtered() -> None:
    centered_bbox = [550.0, 420.0, 850.0, 520.0]

    assert not SpeciesNetDetector._is_edge_anchored_generic_reptile(
        "reptile", centered_bbox, (1080, 1920, 3)
    )
