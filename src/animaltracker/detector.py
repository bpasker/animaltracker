"""Detection backends: YOLO and SpeciesNet."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


class DetectorBackend(str, Enum):
    YOLO = "yolo"
    SPECIESNET = "speciesnet"
    MEGADETECTOR = "megadetector"  # SpeciesNet detect-only mode (fast)


@dataclass
class Detection:
    """Unified detection result across all backends."""
    species: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] pixel coordinates
    taxonomy: Optional[str] = None  # For SpeciesNet: full taxonomy path


class BaseDetector(ABC):
    """Abstract base class for detection backends."""
    
    @abstractmethod
    def infer(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        """Run inference on a single frame."""
        pass
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend identifier."""
        pass


# ============================================================================
# YOLO Backend
# ============================================================================

class YoloDetector(BaseDetector):
    """YOLOv8 detection backend using Ultralytics."""
    
    def __init__(
        self, 
        model_path: str = "yolov8n.pt", 
        class_map: dict[int, str] | None = None
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            raise RuntimeError(
                "Ultralytics YOLO not installed - run: pip install ultralytics"
            )
        
        self.model = YOLO(model_path)
        self.class_map = class_map or self.model.names
        LOGGER.info(f"Loaded YOLO model from {model_path}")

    @property
    def backend_name(self) -> str:
        return "yolo"

    def infer(self, frame: np.ndarray, conf_threshold: float = 0.5, generic_confidence: float = None) -> List[Detection]:
        """Run YOLO inference on a frame.
        
        Args:
            frame: Input image as numpy array
            conf_threshold: Minimum confidence threshold
            generic_confidence: Ignored for YOLO (only used by SpeciesNet)
            
        Returns:
            List of Detection objects
        """
        results = self.model.predict(source=frame, conf=conf_threshold, verbose=False)
        detections: List[Detection] = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                species = self.class_map.get(cls, f"class_{cls}")
                detections.append(Detection(
                    species=species, 
                    confidence=conf, 
                    bbox=bbox
                ))
        
        return detections


# ============================================================================
# MegaDetector Backend (SpeciesNet detect-only mode)
# ============================================================================

class MegaDetectorBackend(BaseDetector):
    """MegaDetector via SpeciesNet's detect-only mode.
    
    Uses SpeciesNet's detection component (MegaDetector v5) without the
    slower species classification step. Ideal for real-time PTZ tracking
    where you need fast bounding boxes but don't need species labels.
    
    MegaDetector categories:
        1 = animal
        2 = person  
        3 = vehicle
    
    Performance: ~100-150ms per frame (vs ~300-500ms for full SpeciesNet)
    """
    
    # MegaDetector category mapping
    CATEGORY_MAP = {
        1: "animal",
        2: "person",
        3: "vehicle",
    }
    
    def __init__(
        self,
        model_version: str = "v4.0.2a",
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize MegaDetector backend.
        
        Args:
            model_version: Model version (v4.0.2a or v4.0.2b)
            cache_dir: Directory for model weights cache
        """
        try:
            from speciesnet import SpeciesNet  # type: ignore
        except ImportError:
            raise RuntimeError(
                "SpeciesNet not installed - run: pip install speciesnet"
            )
        
        self.model_version = model_version
        
        # Initialize SpeciesNet model
        LOGGER.info(f"Loading MegaDetector (SpeciesNet {model_version} detect-only)...")
        model_name = f"kaggle:google/speciesnet/pyTorch/{model_version}/1"
        self._model = SpeciesNet(model_name)
        LOGGER.info("MegaDetector loaded (detect-only mode for fast real-time tracking)")
    
    @property
    def backend_name(self) -> str:
        return "megadetector"

    def infer(
        self, 
        frame: np.ndarray, 
        conf_threshold: float = 0.5, 
        generic_confidence: float = None,  # Ignored for MegaDetector
        return_filtered: bool = False,  # For API compatibility with SpeciesNetBackend
    ) -> List[Detection]:
        """Run MegaDetector inference on a frame.
        
        Args:
            frame: Input image as numpy array
            conf_threshold: Minimum confidence threshold
            generic_confidence: Ignored (MegaDetector only outputs "animal")
            return_filtered: If True, returns (detections, filtered_list) tuple.
                            MegaDetector doesn't filter, so filtered_list is always empty.
        
        Returns:
            List of Detection objects with species="animal" and bounding boxes,
            or tuple (detections, []) if return_filtered=True
        """
        import tempfile
        import cv2
        import os
        
        # SpeciesNet expects file paths, write to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, frame)
        
        try:
            # Run detect-only (skips slow classification)
            result = self._model.detect(
                filepaths=[tmp_path],
                run_mode='multi_thread',
            )
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        detections: List[Detection] = []
        
        if result is None:
            if return_filtered:
                return detections, []
            return detections
        
        predictions_list = result.get("predictions", [])
        
        for pred in predictions_list:
            if not isinstance(pred, dict):
                continue
            
            raw_detections = pred.get("detections", [])
            h, w = frame.shape[:2]
            
            for det in raw_detections:
                conf = det.get("conf", 0.0)
                if conf < conf_threshold:
                    continue
                
                category = det.get("category", 1)
                species = self.CATEGORY_MAP.get(category, "animal")
                
                # Skip non-animal detections for wildlife tracking
                if species != "animal":
                    continue
                
                # Convert bbox from normalized [x, y, w, h] to pixel [x1, y1, x2, y2]
                bbox_norm = det.get("bbox", [0, 0, 1, 1])
                if len(bbox_norm) == 4:
                    bx, by, bw, bh = bbox_norm
                    bbox = [
                        bx * w,
                        by * h,
                        (bx + bw) * w,
                        (by + bh) * h
                    ]
                else:
                    bbox = [0, 0, w, h]
                
                detections.append(Detection(
                    species=species,
                    confidence=conf,
                    bbox=bbox,
                ))
        
        if return_filtered:
            return detections, []  # MegaDetector doesn't filter species
        return detections


# ============================================================================
# SpeciesNet Backend
# ============================================================================

# Regional species blocklists - species that are definitely NOT in these regions
# SpeciesNet's geofencing isn't always complete, so this provides additional filtering
# Users can customize via config's `species_blocklist` option
REGIONAL_BLOCKLISTS = {
    # North America (USA, CAN, MEX) - no wild primates, African megafauna, etc.
    "north_america": {
        # Primates (no wild primates in North America)
        "gibbon", "hylobatidae", "primate", "primates", "ape", "monkey", "chimpanzee", "gorilla",
        "orangutan", "baboon", "macaque", "lemur", "marmoset", "tamarin", "spider_monkey",
        "howler_monkey", "capuchin",
        # African megafauna
        "elephant", "lion", "leopard", "cheetah", "hyena", "zebra", "giraffe", "hippopotamus",
        "rhinoceros", "warthog", "wildebeest", "gnu", "impala", "springbok", "kudu", "oryx",
        "african_buffalo", "cape_buffalo", "aardvark", "pangolin", "okapi", "serval", "caracal",
        # Asian megafauna
        "tiger", "asian_elephant", "gaur", "banteng", "sambar", "chital", "nilgai", "blackbuck",
        "water_buffalo", "yak", "takin", "serow", "goral", "giant_panda",
        "clouded_leopard", "snow_leopard", "dhole", "sloth_bear", "sun_bear", "asian_black_bear",
        # Australian animals
        "kangaroo", "wallaby", "koala", "wombat", "platypus", "echidna", "tasmanian_devil",
        "cassowary", "emu",
        # South American (not in North America proper)
        "jaguar", "tapir", "capybara", "anteater", "sloth", "llama", "alpaca", "vicuna", 
        "guanaco", "mara", "chinchilla",
    },
    # Europe - different set of impossible species
    "europe": {
        "kangaroo", "wallaby", "koala", "platypus", "echidna", "tasmanian_devil",
        "lion", "elephant", "giraffe", "zebra", "hippopotamus", "rhinoceros",
        "tiger", "giant_panda", "gibbon", "orangutan", "gorilla", "chimpanzee",
        "capybara", "tapir", "jaguar", "anteater", "sloth",
    },
    # Australia - filter out non-Australian species often misidentified
    "australia": {
        "lion", "tiger", "leopard", "cheetah", "elephant", "giraffe", "zebra",
        "gorilla", "chimpanzee", "orangutan", "gibbon", "rhinoceros", "hippopotamus",
    },
}

# Map country codes to regional blocklists
COUNTRY_TO_REGION = {
    "USA": "north_america", "CAN": "north_america", "MEX": "north_america",
    "GBR": "europe", "DEU": "europe", "FRA": "europe", "ITA": "europe", 
    "ESP": "europe", "POL": "europe", "NLD": "europe", "BEL": "europe",
    "AUT": "europe", "CHE": "europe", "SWE": "europe", "NOR": "europe",
    "FIN": "europe", "DNK": "europe", "IRL": "europe", "PRT": "europe",
    "AUS": "australia",
}


class SpeciesNetDetector(BaseDetector):
    """Google SpeciesNet detection backend for wildlife camera traps.
    
    SpeciesNet is an ensemble of MegaDetector (object detection) and a 
    species classifier trained on 65M+ camera trap images, supporting 
    2000+ species with geographic filtering.
    """
    
    def __init__(
        self,
        model_version: str = "v4.0.2a",
        country: Optional[str] = None,
        admin1_region: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        generic_confidence: float = 0.9,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize SpeciesNet detector.
        
        Args:
            model_version: Model version (v4.0.2a = crop, v4.0.2b = full-image)
            country: ISO 3166-1 alpha-3 country code for geofencing (e.g., "USA")
            admin1_region: State/province code for US (e.g., "TX")
            latitude: Camera latitude for species range filtering (-90 to 90)
            longitude: Camera longitude for species range filtering (-180 to 180)
            generic_confidence: Higher threshold for generic categories (animal, bird)
            cache_dir: Directory for model weights cache
        """
        try:
            from speciesnet import SpeciesNet  # type: ignore
        except ImportError:
            raise RuntimeError(
                "SpeciesNet not installed - run: pip install speciesnet"
            )
        
        self.country = country
        self.admin1_region = admin1_region
        self.latitude = latitude
        self.longitude = longitude
        self.generic_confidence = generic_confidence
        self.model_version = model_version
        
        # Initialize SpeciesNet model (downloads weights automatically from Kaggle)
        LOGGER.info(f"Loading SpeciesNet {model_version}...")
        model_name = f"kaggle:google/speciesnet/pyTorch/{model_version}/1"
        self._model = SpeciesNet(model_name)
        
        location_info = []
        if country:
            location_info.append(f"country={country}")
        if admin1_region:
            location_info.append(f"region={admin1_region}")
        if latitude is not None and longitude is not None:
            location_info.append(f"coords=({latitude:.4f}, {longitude:.4f})")
        
        loc_str = ", ".join(location_info) if location_info else "no location priors"
        LOGGER.info(f"SpeciesNet loaded ({loc_str}, generic_conf={generic_confidence})")
    
    @property
    def backend_name(self) -> str:
        return "speciesnet"

    def infer(
        self, 
        frame: np.ndarray, 
        conf_threshold: float = 0.5, 
        generic_confidence: float = None,
        return_filtered: bool = False,
    ) -> List[Detection]:
        """Run SpeciesNet inference on a frame.
        
        Args:
            frame: Input image as numpy array
            conf_threshold: Minimum confidence for specific species detections
            generic_confidence: Higher threshold for generic categories (animal, bird).
                               If None, uses the detector's default generic_confidence.
            return_filtered: If True, also returns filtered detections with reason.
                            Returns (detections, filtered_list) tuple instead.
        
        Note: SpeciesNet is optimized for batch processing of images.
        For real-time streaming, consider batching frames or using 
        detector-only mode for speed.
        
        Note: latitude/longitude are stored but not currently passed to predict()
        as the Python API only supports country/admin1_region for geofencing.
        The lat/long would be used for batch JSON input format.
        """
        # Track filtered detections if requested
        filtered_detections: List[Tuple[Detection, str]] = []  # (detection, reason)
        
        # Use instance default if not specified
        if generic_confidence is None:
            generic_confidence = self.generic_confidence
        import tempfile
        import cv2
        
        # SpeciesNet expects file paths, so we write to a temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, frame)
        
        try:
            # Build prediction request with location priors
            # Note: Python API only supports country/admin1_region, not lat/long directly
            result = self._model.predict(
                filepaths=[tmp_path],
                country=self.country,
                admin1_region=self.admin1_region,
            )
        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        detections: List[Detection] = []
        
        # Result is a dict like {"predictions": [{"filepath": ..., "prediction": ..., ...}]}
        if result is None:
            return detections
            
        predictions_list = result.get("predictions", []) if isinstance(result, dict) else []
        
        # Generic categories that require higher confidence
        GENERIC_CATEGORIES = {
            "animal", "bird", "mammalia", "mammal", "aves", 
            "reptilia", "amphibia", "carnivora", "rodentia",
            "passeriformes", "artiodactyla"
        }
        
        for pred in predictions_list:
            if not isinstance(pred, dict):
                continue
                
            # Skip failures
            if pred.get("failures"):
                LOGGER.warning(f"SpeciesNet prediction failed: {pred['failures']}")
                continue
            
            # Get final ensemble prediction
            species = pred.get("prediction", "unknown")
            score = pred.get("prediction_score", 0.0)
            
            # Debug: Log raw SpeciesNet output to understand classifier results
            classifier_output = pred.get("classifier_output", {})
            top_k = classifier_output.get("top_k_prediction", []) if classifier_output else []
            detections_info = pred.get("detections", [])
            det_count = len(detections_info) if detections_info else 0
            LOGGER.debug(
                "SpeciesNet raw: prediction='%s' (%.2f), detections=%d, top_k=%s",
                species, score, det_count, 
                [(p.get("prediction", "?"), p.get("score", 0)) for p in top_k[:3]] if top_k else "none"
            )
            
            # Determine if this is a generic or specific classification
            species_clean = species.lower().strip(";").split(";")[-1].strip().replace(" ", "_")
            
            # Check if any part of the taxonomy is generic (not just the final label)
            taxonomy_parts = [p.lower().strip() for p in species.split(";") if p.strip()]
            is_generic = species_clean in GENERIC_CATEGORIES or (
                len(taxonomy_parts) <= 3 and any(p in GENERIC_CATEGORIES for p in taxonomy_parts)
            )
            
            # Apply tiered confidence threshold
            required_conf = generic_confidence if is_generic else conf_threshold
            if score < required_conf:
                if is_generic:
                    LOGGER.debug("Skipping generic '%s' (%.2f < %.2f generic threshold)", 
                                species_clean, score, required_conf)
                continue
            
            # Skip blanks or non-animal categories
            skip_terms = ("blank", "unknown", "empty", "vehicle", "", "no_cv_result", "no cv result")
            if species_clean in skip_terms or "no cv result" in species.lower():
                continue
            
            # Extract bounding box from detections if available
            bbox = [0.0, 0.0, 1.0, 1.0]  # Default to full frame
            raw_detections = pred.get("detections", [])
            if raw_detections:
                # Use the highest confidence detection bbox
                top_det = max(raw_detections, key=lambda d: d.get("conf", 0))
                # SpeciesNet bbox format: [xmin, ymin, width, height] normalized
                if "bbox" in top_det:
                    bx, by, bw, bh = top_det["bbox"]
                    # Convert to [x1, y1, x2, y2] pixel format
                    h, w = frame.shape[:2]
                    bbox = [
                        bx * w,
                        by * h,
                        (bx + bw) * w,
                        (by + bh) * h
                    ]
            
            # Map common SpeciesNet labels to simpler names
            display_species = self._simplify_species_name(species)
            
            # Double-check: skip blank/unknown/generic after simplification too
            skip_display = ("blank", "unknown", "empty", "no cv result", "no_cv_result")
            display_lower = display_species.lower()
            if display_lower in skip_display or "no cv result" in display_lower:
                if return_filtered:
                    filtered_detections.append((Detection(
                        species=display_species, confidence=score, bbox=bbox, taxonomy=species
                    ), "blank/unknown"))
                continue
            # Also skip if it looks like a UUID (wasn't properly cleaned)
            if len(display_species) > 30 and "-" in display_species and display_species.count("-") >= 3:
                LOGGER.debug("Skipping UUID-like species: %s", display_species[:50])
                if return_filtered:
                    filtered_detections.append((Detection(
                        species=display_species, confidence=score, bbox=bbox, taxonomy=species
                    ), "UUID-like identifier"))
                continue
            
            # Geographic filter: reject species impossible for configured region
            if self._is_exotic_species(display_species, species):
                LOGGER.debug("Filtering exotic species '%s' (impossible in %s)", 
                            display_species, self.country)
                if return_filtered:
                    filtered_detections.append((Detection(
                        species=display_species, confidence=score, bbox=bbox, taxonomy=species
                    ), f"exotic species (impossible in {self.country})"))
                continue
            
            detections.append(Detection(
                species=display_species,
                confidence=score,
                bbox=bbox,
                taxonomy=species,  # Keep full taxonomy
            ))
        
        if return_filtered:
            return detections, filtered_detections
        return detections
    
    def _is_exotic_species(self, display_name: str, taxonomy: str) -> bool:
        """Check if a species is impossible for the configured region.
        
        Uses regional blocklists based on country setting. SpeciesNet's
        geofencing isn't always complete, so this provides extra filtering.
        """
        if not self.country:
            return False  # No filtering if no location set
        
        # Get blocklist for this region
        region = COUNTRY_TO_REGION.get(self.country)
        if not region:
            return False  # Country not in our regional mappings
        
        blocklist = REGIONAL_BLOCKLISTS.get(region, set())
        if not blocklist:
            return False
        
        # Check display name
        display_lower = display_name.lower().replace(" ", "_").replace("-", "_")
        for exotic in blocklist:
            if exotic in display_lower:
                return True
        
        # Check taxonomy parts (handles "mammalia_primates_hylobatidae")
        taxonomy_lower = taxonomy.lower().replace(" ", "_").replace("-", "_")
        for exotic in blocklist:
            if exotic in taxonomy_lower:
                return True
        
        return False
    
    def _simplify_species_name(self, taxonomy: str) -> str:
        """Convert taxonomy label to display-friendly name.
        
        SpeciesNet returns complex labels like:
        - 'odocoileus_virginianus' (scientific name)
        - '1F689929-...;;;;;;Animal' (UUID + hierarchy)
        - 'B1352069-...;Aves;;;;;Bird' (UUID + taxonomy path)
        
        This extracts clean, human-readable names.
        """
        import re
        
        SPECIES_MAP = {
            # Deer
            "odocoileus_virginianus": "whitetail deer",
            "odocoileus_hemionus": "mule deer",
            "cervus_elaphus": "elk",
            "alces_alces": "moose",
            # Bears
            "ursus_americanus": "black bear",
            "ursus_arctos": "grizzly bear",
            # Cats
            "felis_catus": "cat",
            "puma_concolor": "mountain lion",
            "lynx_rufus": "bobcat",
            "lynx_canadensis": "lynx",
            # Dogs/Canids
            "canis_familiaris": "dog",
            "canis_latrans": "coyote",
            "canis_lupus": "wolf",
            "vulpes_vulpes": "red fox",
            "urocyon_cinereoargenteus": "gray fox",
            # Other mammals
            "procyon_lotor": "raccoon",
            "didelphis_virginiana": "opossum",
            "mephitis_mephitis": "skunk",
            "sylvilagus": "rabbit",
            "sylvilagus_floridanus": "cottontail rabbit",
            "lepus": "hare",
            "sciurus": "squirrel",
            "sciurus_carolinensis": "gray squirrel",
            "sus_scrofa": "wild boar",
            "pecari_tajacu": "javelina",
            "nasua_narica": "coati",
            "taxidea_taxus": "badger",
            "lontra_canadensis": "river otter",
            "mustela": "weasel",
            "neovison_vison": "mink",
            "castor_canadensis": "beaver",
            "erethizon_dorsatum": "porcupine",
            "marmota": "marmot",
            # Birds - common species
            "meleagris_gallopavo": "wild turkey",
            "aves": "bird",
            "bird": "bird",
            "corvus_brachyrhynchos": "american crow",
            "corvus_corax": "common raven",
            "haliaeetus_leucocephalus": "bald eagle",
            "buteo_jamaicensis": "red-tailed hawk",
            "buteo": "hawk",
            "accipiter": "hawk",
            "strix_varia": "barred owl",
            "bubo_virginianus": "great horned owl",
            "megascops": "screech owl",
            "zenaida_macroura": "mourning dove",
            "branta_canadensis": "canada goose",
            "anas_platyrhynchos": "mallard duck",
            "ardea_herodias": "great blue heron",
            "cathartes_aura": "turkey vulture",
            "coragyps_atratus": "black vulture",
            "phasianus_colchicus": "pheasant",
            "colinus_virginianus": "bobwhite quail",
            "geococcyx_californianus": "roadrunner",
            # Humans/vehicles
            "homo_sapiens": "person",
            "human": "person",
            "animal": "animal",
            "vehicle": "vehicle",
        }
        
        # Check direct match first
        tax_lower = taxonomy.lower()
        if tax_lower in SPECIES_MAP:
            return SPECIES_MAP[tax_lower]
        
        # Handle SpeciesNet's complex taxonomy format with UUIDs and semicolons
        # Format: UUID;Class;Order;Family;Genus;Species;CommonName
        # Example: "1F689929-...;Aves;Passeriformes;Corvidae;Corvus;corvus_brachyrhynchos;American Crow"
        # Or generic: "UUID;Aves;;;;;Bird"
        
        # Split by + to handle multiple detections
        parts = taxonomy.split("+")
        clean_names = []
        
        for part in parts:
            # Remove UUID prefix (8-4-4-4-12 hex pattern)
            part = re.sub(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}[;]*', '', part)
            
            # Split by semicolons - taxonomy levels are: Class;Order;Family;Genus;Species;CommonName
            segments = [s.strip() for s in part.split(";")]
            
            # Filter out empty and useless values
            meaningful = []
            for seg in segments:
                seg_lower = seg.lower()
                if seg_lower in ("no cv result", "unknown", "blank", "empty", ""):
                    continue
                # Skip if it looks like a UUID
                if re.match(r'^[0-9a-fA-F-]+$', seg) and len(seg) > 10:
                    continue
                meaningful.append(seg)
            
            if not meaningful:
                continue
            
            # Build a taxonomy string with available levels
            # Priority: use common name if specific, otherwise build from class/order/family
            taxonomy_parts = []
            
            # Check if we have a specific species (last meaningful segment that's in SPECIES_MAP or looks specific)
            last_seg = meaningful[-1].lower()
            if last_seg in SPECIES_MAP:
                # Use the friendly name from our map
                taxonomy_parts.append(SPECIES_MAP[last_seg])
            else:
                # Build from available taxonomy levels
                for seg in meaningful:
                    seg_lower = seg.lower()
                    # Map known classes/orders to friendly names or keep as-is
                    if seg_lower in SPECIES_MAP:
                        taxonomy_parts.append(SPECIES_MAP[seg_lower])
                    elif len(seg) > 1:
                        clean_seg = seg.replace("_", "-").lower()
                        # Only add if not redundant
                        if clean_seg not in [t.lower() for t in taxonomy_parts]:
                            taxonomy_parts.append(clean_seg)
            
            if taxonomy_parts:
                # Join with underscore for filename compatibility, limit to 3 levels
                name = "_".join(taxonomy_parts[:3])
                if name not in clean_names:
                    clean_names.append(name)
        
        if clean_names:
            # Return names joined by + for multiple detections
            return "+".join(clean_names)
        
        # Fallback: check if it starts with a known genus
        for key, value in SPECIES_MAP.items():
            if tax_lower.startswith(key.split("_")[0]):
                return value
        
        # Last resort: clean up and return
        return taxonomy.replace("_", "-").lower()[:50]  # Truncate if too long


# ============================================================================
# Detector Factory
# ============================================================================

def create_detector(
    backend: DetectorBackend | str = DetectorBackend.YOLO,
    **kwargs
) -> BaseDetector:
    """Factory function to create a detector instance.
    
    Args:
        backend: Which detection backend to use ("yolo", "speciesnet", or "megadetector")
        **kwargs: Backend-specific configuration
        
    YOLO kwargs:
        - model_path: Path to YOLO weights (default: "yolov8n.pt")
        - class_map: Optional class ID to name mapping
        
    MegaDetector kwargs:
        - model_version: "v4.0.2a" (crop) or "v4.0.2b" (full-image)
        
    SpeciesNet kwargs:
        - model_version: "v4.0.2a" (crop) or "v4.0.2b" (full-image)
        - country: ISO 3166-1 alpha-3 code (e.g., "USA")
        - admin1_region: State code for US (e.g., "TX")
        - latitude: Camera latitude for species range filtering
        - longitude: Camera longitude for species range filtering
        - generic_confidence: Higher threshold for generic categories (default 0.9)
        - cache_dir: Model weights cache directory
    
    Returns:
        Configured detector instance
    """
    if isinstance(backend, str):
        backend = DetectorBackend(backend.lower())
    
    if backend == DetectorBackend.YOLO:
        return YoloDetector(
            model_path=kwargs.get("model_path", "yolov8n.pt"),
            class_map=kwargs.get("class_map"),
        )
    
    elif backend == DetectorBackend.MEGADETECTOR:
        return MegaDetectorBackend(
            model_version=kwargs.get("model_version", "v4.0.2a"),
            cache_dir=kwargs.get("cache_dir"),
        )
    
    elif backend == DetectorBackend.SPECIESNET:
        return SpeciesNetDetector(
            model_version=kwargs.get("model_version", "v4.0.2a"),
            country=kwargs.get("country"),
            admin1_region=kwargs.get("admin1_region"),
            latitude=kwargs.get("latitude"),
            longitude=kwargs.get("longitude"),
            generic_confidence=kwargs.get("generic_confidence", 0.9),
            cache_dir=kwargs.get("cache_dir"),
        )
    
    else:
        raise ValueError(f"Unknown detector backend: {backend}")


def create_realtime_detector(detector_cfg) -> BaseDetector:
    """Create detector optimized for real-time streaming and PTZ tracking.
    
    Uses the realtime_backend setting (default: YOLO for speed).
    Falls back to legacy 'backend' setting if realtime_backend not specified.
    
    Args:
        detector_cfg: DetectorSettings from config
        
    Returns:
        Fast detector for real-time use (~50-150ms inference)
    """
    backend = getattr(detector_cfg, 'realtime_backend', None) or detector_cfg.backend
    
    LOGGER.info(f"Creating realtime detector: {backend}")
    
    return create_detector(
        backend=backend,
        model_path=detector_cfg.model_path,
        model_version=detector_cfg.speciesnet_version,
        country=detector_cfg.country,
        admin1_region=detector_cfg.admin1_region,
        latitude=detector_cfg.latitude,
        longitude=detector_cfg.longitude,
        generic_confidence=detector_cfg.generic_confidence,
    )


def create_postprocess_detector(detector_cfg) -> BaseDetector:
    """Create detector optimized for post-clip species identification.
    
    Uses the postprocess_backend setting (default: SpeciesNet for accuracy).
    Falls back to legacy 'backend' setting if postprocess_backend not specified.
    
    Args:
        detector_cfg: DetectorSettings from config
        
    Returns:
        Accurate detector for post-processing (~200-500ms inference)
    """
    backend = getattr(detector_cfg, 'postprocess_backend', None) or detector_cfg.backend
    
    LOGGER.info(f"Creating postprocess detector: {backend}")
    
    return create_detector(
        backend=backend,
        model_path=detector_cfg.model_path,
        model_version=detector_cfg.speciesnet_version,
        country=detector_cfg.country,
        admin1_region=detector_cfg.admin1_region,
        latitude=detector_cfg.latitude,
        longitude=detector_cfg.longitude,
        generic_confidence=detector_cfg.generic_confidence,
    )


# Keep backward compatibility
__all__ = [
    "Detection",
    "DetectorBackend", 
    "BaseDetector",
    "YoloDetector",
    "MegaDetectorBackend",
    "SpeciesNetDetector",
    "create_detector",
    "create_realtime_detector",
    "create_postprocess_detector",
]
