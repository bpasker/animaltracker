"""Species name mapping from scientific/technical names to common names.

This module provides human-readable common names for species detected by SpeciesNet.
The mapping is hierarchical - it checks for full matches first, then partial matches.
"""

from typing import Optional
import re

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
    
    # Squirrels
    "mammalia_rodentia_sciuridae": "Squirrel",
    "mammalia_rodentia_sciuridae_sciurus_carolinensis": "Eastern Gray Squirrel",
    "mammalia_rodentia_sciuridae_sciurus_niger": "Fox Squirrel",
    "mammalia_rodentia_sciuridae_tamias_striatus": "Eastern Chipmunk",
    "mammalia_rodentia_sciuridae_tamias": "Chipmunk",
    
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
    (r"sciuridae", "Squirrel"),
    (r"leporidae", "Rabbit"),
    (r"didelphidae", "Opossum"),
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
