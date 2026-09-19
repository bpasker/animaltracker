"""Species name mapping from scientific/technical names to common names.

This module provides human-readable common names for species detected by SpeciesNet.
The mapping is hierarchical - it checks for full matches first, then partial matches.
"""

from typing import Dict, List, Mapping, Optional, Tuple
import re

# Labels SpeciesNet emits for things that are not wildlife. They are never a
# clip's species, but they are real observations: the post-processor keeps
# their boxes so that an animal label sitting exactly where a person was seen
# a moment earlier can be recognised as that person (see postprocess.py).
NON_ANIMAL_LABELS = frozenset({"person", "human", "vehicle"})

# --- How specific a label is, and which labels agree -------------------------
#
# The detector names an animal by the taxonomy levels SpeciesNet was sure of,
# joined with "_": class, order, family ("mammalia_carnivora_canidae"). A
# rollup stops early and ends with its common name instead
# ("mammalia_carnivora_carnivorous mammal", "mammalia_mammal", "bird"), and
# "animal" names no taxon at all. How specific a label is, and whether two
# labels agree, follow from that path alone. Every consumer ranks labels
# through the functions below rather than from keyword lists: a list scores
# the families it happens to name differently from the ones it forgot, and a
# forgotten family then outranks, or loses to, a label at the very same level
# (cardinalidae scored 2, corvidae 3 and phasianidae 4, so one stray crow
# frame beat forty cardinal frames).

# Labels that name no taxon.
_UNRANKED_LABELS = frozenset({"", "animal", "unknown", "blank", "empty"})

# The class level arrives under two spellings, the taxon ("mammalia") and the
# rollup's common name ("mammal"); "aves" is already mapped to "bird" by the
# detector. One token per class keeps a rollup and its families on one branch.
_CLASS_TOKENS = {
    "mammalia": "mammalia", "mammal": "mammalia",
    "aves": "bird", "bird": "bird",
    "reptilia": "reptile", "reptile": "reptile",
    "amphibia": "amphibian", "amphibian": "amphibian",
    "fish": "fish",
}

# Every zoological family name ends this way (ICZN), and no common name does.
_FAMILY_SUFFIX = "idae"


def species_lineage(species: str) -> Tuple[str, ...]:
    """The taxonomy path a species label names, most general level first.

    ``"mammalia_carnivora_canidae"`` is ``("mammalia", "carnivora", "canidae")``;
    the order rollup ``"mammalia_carnivora_carnivorous mammal"`` is
    ``("mammalia", "carnivora")``; ``"mammalia_mammal"`` and ``"mammal"`` are
    ``("mammalia",)``; ``"animal"`` is ``()``. A label is an ancestor of
    another exactly when its lineage is a prefix of the other's, and two
    labels contradict each other when neither is.

    The levels are read by position: class, then order, then a family, which
    is recognised by its suffix. Whatever sits where a taxon should but is
    not one is the rollup's common name and is dropped, so a label with its
    common name and the same label without it have the same lineage. A word
    with no taxonomy at all (YOLO's ``"dog"``) is a branch of its own.
    """
    label = (species or "").strip().lower()
    if "+" in label:
        # Several detections joined into one name: rank it by the first.
        label = label.split("+", 1)[0].strip()
    if label in _UNRANKED_LABELS:
        return ()
    tokens = [t.strip() for t in label.split("_") if t.strip()]
    if not tokens:
        return ()

    head = _CLASS_TOKENS.get(tokens[0], tokens[0])
    lineage: List[str] = [head]
    rest = tokens[1:]
    if not rest or head.endswith(_FAMILY_SUFFIX):
        return tuple(lineage)

    # Order, unless the slot holds the class's own common name
    # ("mammalia_mammal") or the order is missing and a family follows.
    if rest[0].endswith(_FAMILY_SUFFIX):
        lineage.append(rest[0])
        lineage.extend(rest[1:])
        return tuple(lineage)
    if _CLASS_TOKENS.get(rest[0]) == head:
        return tuple(lineage)
    lineage.append(rest[0])

    # Family, or the order rollup's common name ("..._rodentia_rodent").
    if len(rest) > 1 and rest[1].endswith(_FAMILY_SUFFIX):
        lineage.append(rest[1])
        lineage.extend(rest[2:])  # genus, species: not produced today
    return tuple(lineage)


def species_rank(species: str) -> int:
    """How specific a label is: 0 animal, 1 class, 2 order, 3 family, 4+ below.

    Labels at the same taxonomic level always get the same rank, whatever
    family or order they name.
    """
    lineage = species_lineage(species)
    if not lineage:
        return 0
    if len(lineage) < 3 and lineage[-1].endswith(_FAMILY_SUFFIX):
        return 3  # a family named without its class or order
    return len(lineage)


def pick_species_by_lineage(votes: Mapping[str, Tuple[float, float]]) -> str:
    """The label a set of votes agrees on, found by walking down the taxonomy.

    ``votes`` maps each label to ``(count, confidence)``. From the root, the
    walk takes the branch with the most votes (count first, then the best
    confidence) and keeps going while any label lies further down. Two
    things follow:

    * a specific label beats its own generic ancestors however few votes it
      has, because they do not contradict it: five "cervidae" frames among
      fifty "animal" frames are a deer;
    * labels that contradict each other are settled by their votes, at the
      level where they part: forty "cardinalidae" frames beat one "corvidae"
      frame, and forty dog frames beat one turkey frame, whatever the
      confidence of the stray.

    Returns ``""`` for no votes. Exact ties go to the label seen first.
    """
    if not votes:
        return ""
    lineages: Dict[str, Tuple[str, ...]] = {label: species_lineage(label) for label in votes}
    prefix: Tuple[str, ...] = ()
    while True:
        depth = len(prefix)
        branches: Dict[str, List[float]] = {}
        for label, lineage in lineages.items():
            if len(lineage) <= depth or lineage[:depth] != prefix:
                continue
            count, confidence = votes[label]
            branch = branches.setdefault(lineage[depth], [0.0, 0.0])
            branch[0] += count
            branch[1] = max(branch[1], confidence)
        if not branches:
            break
        best = max(branches, key=lambda token: (branches[token][0], branches[token][1]))
        prefix = prefix + (best,)
    at_node = [label for label, lineage in lineages.items() if lineage == prefix]
    return max(at_node, key=lambda label: (votes[label][0], votes[label][1]))


# Mapping from technical names to common names
# Format: lowercase key -> display name
# Keys can be full species names or partial matches (family/genus level)
SPECIES_MAP = {
    # === BIRDS ===
    # Cardinals
    "bird_passeriformes_cardinalidae": "Cardinal",
    "bird_passeriformes_cardinalidae_cardinalis_cardinalis": "Northern Cardinal",
    "bird_passeriformes_cardinalidae_cardinalis": "Cardinal",
    
    # Blue Jays & Corvids
    "bird_passeriformes_corvidae": "Crow/Jay Family",
    "bird_passeriformes_corvidae_cyanocitta_cristata": "Blue Jay",
    "bird_passeriformes_corvidae_cyanocitta": "Blue Jay",
    "bird_passeriformes_corvidae_corvus": "Crow",
    "bird_passeriformes_corvidae_corvus_brachyrhynchos": "American Crow",
    
    # Sparrows
    "bird_passeriformes_passerellidae": "Sparrow",
    "bird_passeriformes_passerellidae_melospiza_melodia": "Song Sparrow",
    "bird_passeriformes_passerellidae_zonotrichia_albicollis": "White-throated Sparrow",
    
    # Finches
    "bird_passeriformes_fringillidae": "Finch",
    "bird_passeriformes_fringillidae_haemorhous_mexicanus": "House Finch",
    "bird_passeriformes_fringillidae_spinus_tristis": "American Goldfinch",
    
    # Thrushes (Robins)
    "bird_passeriformes_turdidae": "Thrush",
    "bird_passeriformes_turdidae_turdus_migratorius": "American Robin",
    "bird_passeriformes_turdidae_turdus": "Robin/Thrush",
    
    # Mockingbirds
    "bird_passeriformes_mimidae": "Mockingbird Family",
    "bird_passeriformes_mimidae_mimus_polyglottos": "Northern Mockingbird",
    
    # Woodpeckers
    "bird_piciformes_picidae": "Woodpecker",
    "bird_piciformes_picidae_melanerpes_carolinus": "Red-bellied Woodpecker",
    "bird_piciformes_picidae_dryobates_pubescens": "Downy Woodpecker",
    
    # Hummingbirds
    "bird_apodiformes_trochilidae": "Hummingbird",
    "bird_apodiformes_trochilidae_archilochus_colubris": "Ruby-throated Hummingbird",
    
    # Doves/Pigeons
    "bird_columbiformes_columbidae": "Dove/Pigeon",
    "bird_columbiformes_columbidae_zenaida_macroura": "Mourning Dove",
    "bird_columbiformes_columbidae_columba_livia": "Rock Pigeon",
    
    # Hawks/Eagles
    "bird_accipitriformes_accipitridae": "Hawk/Eagle",
    "bird_accipitriformes_accipitridae_buteo_jamaicensis": "Red-tailed Hawk",
    "bird_accipitriformes_accipitridae_accipiter_cooperii": "Cooper's Hawk",
    
    # Owls
    "bird_strigiformes_strigidae": "Owl",
    "bird_strigiformes_strigidae_bubo_virginianus": "Great Horned Owl",
    "bird_strigiformes_strigidae_megascops_asio": "Eastern Screech Owl",
    
    # Wrens
    "bird_passeriformes_troglodytidae": "Wren",
    "bird_passeriformes_troglodytidae_troglodytes_aedon": "House Wren",
    
    # Chickadees/Titmice
    "bird_passeriformes_paridae": "Chickadee/Titmouse",
    "bird_passeriformes_paridae_poecile_carolinensis": "Carolina Chickadee",
    "bird_passeriformes_paridae_baeolophus_bicolor": "Tufted Titmouse",
    
    # Nuthatches
    "bird_passeriformes_sittidae": "Nuthatch",
    "bird_passeriformes_sittidae_sitta_carolinensis": "White-breasted Nuthatch",
    
    # Blackbirds/Grackles
    "bird_passeriformes_icteridae": "Blackbird",
    "bird_passeriformes_icteridae_quiscalus": "Grackle",
    "bird_passeriformes_icteridae_quiscalus_quiscula": "Common Grackle",
    "bird_passeriformes_icteridae_agelaius_phoeniceus": "Red-winged Blackbird",
    
    # Warblers
    "bird_passeriformes_parulidae": "Warbler",
    
    # Generic bird categories
    "bird_passeriformes": "Songbird",
    "bird": "Bird",
    
    # === MAMMALS ===
    # Cats
    "mammalia_carnivora_felidae": "Cat",
    "mammalia_carnivora_felidae_felis_catus": "Domestic Cat",
    "mammalia_carnivora_felidae_lynx_rufus": "Bobcat",
    
    # Dogs/Canids
    "mammalia_carnivora_canidae": "Dog/Canid",
    "mammalia_carnivora_canidae_canis_familiaris": "Domestic Dog",
    "mammalia_carnivora_canidae_canis_latrans": "Coyote",
    "mammalia_carnivora_canidae_vulpes_vulpes": "Red Fox",
    "mammalia_carnivora_canidae_urocyon_cinereoargenteus": "Gray Fox",
    
    # Raccoons
    "mammalia_carnivora_procyonidae": "Raccoon Family",
    "mammalia_carnivora_procyonidae_procyon_lotor": "Raccoon",
    "mammalia_carnivora_procyonidae_procyon": "Raccoon",
    
    # Skunks
    "mammalia_carnivora_mephitidae": "Skunk",
    "mammalia_carnivora_mephitidae_mephitis_mephitis": "Striped Skunk",
    
    # Bears
    "mammalia_carnivora_ursidae": "Bear",
    "mammalia_carnivora_ursidae_ursus_americanus": "Black Bear",
    
    # Deer
    "mammalia_artiodactyla_cervidae": "Deer",
    "mammalia_artiodactyla_cervidae_odocoileus_virginianus": "White-tailed Deer",
    "mammalia_artiodactyla_cervidae_odocoileus": "Deer",
    
    # Bovids (cattle, goats, sheep, bison, antelopes)
    "mammalia_artiodactyla_bovidae": "Bovid (Cattle/Goat/Sheep)",
    "mammalia_artiodactyla_bovidae_bos_taurus": "Cattle",
    "mammalia_artiodactyla_bovidae_bos": "Cattle",
    "mammalia_artiodactyla_bovidae_capra": "Goat",
    "mammalia_artiodactyla_bovidae_capra_hircus": "Domestic Goat",
    "mammalia_artiodactyla_bovidae_ovis": "Sheep",
    "mammalia_artiodactyla_bovidae_ovis_aries": "Domestic Sheep",
    "mammalia_artiodactyla_bovidae_bison": "Bison",
    "mammalia_artiodactyla_bovidae_bison_bison": "American Bison",
    
    # Squirrels
    "mammalia_rodentia_sciuridae": "Squirrel",
    "mammalia_rodentia_sciuridae_sciurus_carolinensis": "Eastern Gray Squirrel",
    "mammalia_rodentia_sciuridae_sciurus_niger": "Fox Squirrel",
    "mammalia_rodentia_sciuridae_tamias_striatus": "Eastern Chipmunk",
    "mammalia_rodentia_sciuridae_tamias": "Chipmunk",
    
    # Generic rodent (order level)
    "mammalia_rodentia_rodent": "Rodent",
    "mammalia_rodentia": "Rodent",
    
    # Rabbits
    "mammalia_lagomorpha_leporidae": "Rabbit",
    "mammalia_lagomorpha_leporidae_sylvilagus_floridanus": "Eastern Cottontail",
    "mammalia_lagomorpha_leporidae_sylvilagus": "Cottontail Rabbit",
    
    # Opossums
    "mammalia_didelphimorphia_didelphidae": "Opossum",
    "mammalia_didelphimorphia_didelphidae_didelphis_virginiana": "Virginia Opossum",
    
    # Armadillos
    "mammalia_cingulata_dasypodidae": "Armadillo",
    "mammalia_cingulata_dasypodidae_dasypus_novemcinctus": "Nine-banded Armadillo",
    
    # Groundhogs/Woodchucks
    "mammalia_rodentia_sciuridae_marmota_monax": "Groundhog",
    
    # Primates (probably misclassifications in North America!)
    "mammalia_primates_hylobatidae": "Gibbon (likely misidentified)",
    "mammalia_primates": "Primate",
    
    # Generic carnivore
    "mammalia_carnivora_carnivorous_mammal": "Carnivore (Cat/Dog/Raccoon)",
    "mammalia_carnivora": "Carnivore",
    
    # Generic mammal
    "mammalia": "Mammal",
    
    # === REPTILES ===
    "reptilia_squamata_colubridae": "Snake (Colubrid)",
    "reptilia_squamata_viperidae": "Venomous Snake",
    "reptilia_testudines": "Turtle/Tortoise",
    "reptilia": "Reptile",
    
    # === GENERIC ===
    "animal": "Animal",
    "unknown": "Unknown",
    "blank": "Empty Frame",
}

# Partial match patterns - checked when full match fails
# These are regex patterns matched against the normalized species name
PARTIAL_PATTERNS = [
    (r"cardinalidae", "Cardinal"),
    (r"corvidae", "Crow/Jay"),
    (r"passerellidae", "Sparrow"),
    (r"fringillidae", "Finch"),
    (r"turdidae", "Robin/Thrush"),
    (r"picidae", "Woodpecker"),
    (r"trochilidae", "Hummingbird"),
    (r"columbidae", "Dove/Pigeon"),
    (r"accipitridae", "Hawk/Eagle"),
    (r"strigidae", "Owl"),
    (r"felidae", "Cat"),
    (r"canidae", "Dog/Canid"),
    (r"procyonidae", "Raccoon"),
    (r"cervidae", "Deer"),
    (r"bovidae", "Bovid (Cattle/Goat/Sheep)"),
    (r"sciuridae", "Squirrel"),
    (r"leporidae", "Rabbit"),
    (r"didelphidae", "Opossum"),
    (r"rodentia", "Rodent"),
    (r"carnivora", "Carnivore"),
    (r"artiodactyla", "Hoofed Mammal"),
    (r"lagomorpha", "Rabbit"),
]


def get_common_name(species: str) -> str:
    """Get the common name for a species.
    
    Args:
        species: The scientific/technical species name (e.g., "bird_passeriformes_cardinalidae")
        
    Returns:
        Human-readable common name (e.g., "Cardinal")
    """
    if not species:
        return "Unknown"
    
    # Normalize the input
    normalized = species.lower().replace(" ", "_").replace("-", "_")
    
    # Try exact match first
    if normalized in SPECIES_MAP:
        return SPECIES_MAP[normalized]
    
    # Try progressively shorter prefixes (most specific to least)
    parts = normalized.split("_")
    for i in range(len(parts), 0, -1):
        prefix = "_".join(parts[:i])
        if prefix in SPECIES_MAP:
            return SPECIES_MAP[prefix]
    
    # Try partial pattern matching
    for pattern, name in PARTIAL_PATTERNS:
        if re.search(pattern, normalized):
            return name
    
    # Fall back to title-casing the last meaningful part
    for part in reversed(parts):
        if part and part not in ('animal', 'bird', 'mammalia', 'unknown', 'blank'):
            return part.replace("_", " ").title()
    
    # Last resort - just title case the whole thing
    return species.replace("_", " ").title()


def format_species_display(species: str, include_scientific: bool = False) -> str:
    """Format a species name for display.
    
    Args:
        species: The scientific/technical species name
        include_scientific: If True, include scientific name in parentheses
        
    Returns:
        Formatted display string
    """
    common = get_common_name(species)
    
    if include_scientific and common.lower() != species.lower().replace("_", " "):
        # Clean up scientific name for display
        scientific = species.replace("_", " ").title()
        return f"{common} ({scientific})"
    
    return common


def add_custom_mapping(technical_name: str, common_name: str) -> None:
    """Add a custom species mapping at runtime.
    
    This can be used to extend the mapping based on user preferences
    or local wildlife.
    
    Args:
        technical_name: The technical/scientific name (will be normalized)
        common_name: The human-readable common name
    """
    normalized = technical_name.lower().replace(" ", "_").replace("-", "_")
    SPECIES_MAP[normalized] = common_name


def get_species_icon(species: str) -> str:
    """Get an emoji icon appropriate for the species type.
    
    Args:
        species: The scientific/technical species name
        
    Returns:
        Emoji icon string
    """
    if not species:
        return "❓"
    
    normalized = species.lower().replace(" ", "_").replace("-", "_")
    
    # Check for birds
    if normalized.startswith("bird") or "aves" in normalized:
        return "🐦"
    
    # Check for specific mammal types
    if "felidae" in normalized or "cat" in normalized:
        return "🐱"
    if "canidae" in normalized or "dog" in normalized or "coyote" in normalized or "fox" in normalized:
        return "🐕"
    if "cervidae" in normalized or "deer" in normalized:
        return "🦌"
    if "ursidae" in normalized or "bear" in normalized:
        return "🐻"
    if "procyonidae" in normalized or "raccoon" in normalized:
        return "🦝"
    if "sciuridae" in normalized or "squirrel" in normalized or "chipmunk" in normalized:
        return "🐿️"
    if "leporidae" in normalized or "rabbit" in normalized or "lagomorpha" in normalized:
        return "🐰"
    if "didelphidae" in normalized or "opossum" in normalized:
        return "🐀"
    if "mephitidae" in normalized or "skunk" in normalized:
        return "🦨"
    if "bovidae" in normalized:
        return "🐄"
    if "rodentia" in normalized:
        return "🐭"
    
    # Check for reptiles
    if normalized.startswith("reptilia") or "squamata" in normalized:
        if "colubridae" in normalized or "viperidae" in normalized or "snake" in normalized:
            return "🐍"
        if "testudines" in normalized or "turtle" in normalized:
            return "🐢"
        return "🦎"
    
    # Check for amphibians
    if "amphibia" in normalized or "frog" in normalized or "toad" in normalized:
        return "🐸"
    
    # Check for humans/primates
    if "primates" in normalized or "human" in normalized or "person" in normalized:
        return "🧑"
    
    # Generic mammal
    if normalized.startswith("mammalia") or "mammal" in normalized:
        return "🐾"
    
    # Unknown animal
    if "animal" in normalized or "unknown" in normalized:
        return "❓"
    
    # Default to paw print for any unrecognized animal
    return "🐾"
