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

    def infer(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
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
# SpeciesNet Backend
# ============================================================================

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
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize SpeciesNet detector.
        
        Args:
            model_version: Model version (v4.0.2a = crop, v4.0.2b = full-image)
            country: ISO 3166-1 alpha-3 country code for geofencing (e.g., "USA")
            admin1_region: State/province code for US (e.g., "CA")
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
        self.model_version = model_version
        
        # Initialize SpeciesNet model (downloads weights automatically)
        LOGGER.info(f"Loading SpeciesNet {model_version}...")
        self._model = SpeciesNet(model_name=f"google/speciesnet/pyTorch/{model_version}")
        LOGGER.info(f"SpeciesNet loaded (country={country}, region={admin1_region})")
    
    @property
    def backend_name(self) -> str:
        return "speciesnet"

    def infer(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        """Run SpeciesNet inference on a frame.
        
        Note: SpeciesNet is optimized for batch processing of images.
        For real-time streaming, consider batching frames or using 
        detector-only mode for speed.
        """
        import tempfile
        import cv2
        
        # SpeciesNet expects file paths, so we write to a temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            cv2.imwrite(tmp.name, frame)
            
            # Build prediction request
            predictions = self._model.predict(
                filepaths=[tmp.name],
                country=self.country,
                admin1_region=self.admin1_region,
            )
        
        detections: List[Detection] = []
        
        for pred in predictions:
            # Skip failures
            if pred.get("failures"):
                LOGGER.warning(f"SpeciesNet prediction failed: {pred['failures']}")
                continue
            
            # Get final ensemble prediction
            species = pred.get("prediction", "unknown")
            score = pred.get("prediction_score", 0.0)
            
            # Skip low confidence or blanks
            if score < conf_threshold:
                continue
            if species in ("blank", "unknown"):
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
            
            detections.append(Detection(
                species=display_species,
                confidence=score,
                bbox=bbox,
                taxonomy=species,  # Keep full taxonomy
            ))
        
        return detections
    
    def _simplify_species_name(self, taxonomy: str) -> str:
        """Convert taxonomy label to display-friendly name.
        
        SpeciesNet returns labels like 'odocoileus_virginianus' (white-tailed deer).
        This maps common ones to readable names.
        """
        SPECIES_MAP = {
            # Deer
            "odocoileus_virginianus": "deer",
            "odocoileus_hemionus": "deer",
            "cervus_elaphus": "elk",
            "alces_alces": "moose",
            # Bears
            "ursus_americanus": "bear",
            "ursus_arctos": "bear",
            # Cats
            "felis_catus": "cat",
            "puma_concolor": "mountain_lion",
            "lynx_rufus": "bobcat",
            # Dogs/Canids
            "canis_familiaris": "dog",
            "canis_latrans": "coyote",
            "vulpes_vulpes": "fox",
            "urocyon_cinereoargenteus": "fox",
            # Other mammals
            "procyon_lotor": "raccoon",
            "didelphis_virginiana": "opossum",
            "mephitis_mephitis": "skunk",
            "sylvilagus": "rabbit",
            "sciurus": "squirrel",
            "sus_scrofa": "wild_boar",
            # Birds
            "meleagris_gallopavo": "turkey",
            # Humans/vehicles
            "homo_sapiens": "person",
            "human": "person",
            "vehicle": "vehicle",
        }
        
        # Check direct match
        if taxonomy in SPECIES_MAP:
            return SPECIES_MAP[taxonomy]
        
        # Check if it starts with a known genus
        for key, value in SPECIES_MAP.items():
            if taxonomy.startswith(key.split("_")[0]):
                return value
        
        # Return cleaned up version of taxonomy
        return taxonomy.replace("_", " ")


# ============================================================================
# Detector Factory
# ============================================================================

def create_detector(
    backend: DetectorBackend | str = DetectorBackend.YOLO,
    **kwargs
) -> BaseDetector:
    """Factory function to create a detector instance.
    
    Args:
        backend: Which detection backend to use ("yolo" or "speciesnet")
        **kwargs: Backend-specific configuration
        
    YOLO kwargs:
        - model_path: Path to YOLO weights (default: "yolov8n.pt")
        - class_map: Optional class ID to name mapping
        
    SpeciesNet kwargs:
        - model_version: "v4.0.2a" (crop) or "v4.0.2b" (full-image)
        - country: ISO 3166-1 alpha-3 code (e.g., "USA")
        - admin1_region: State code for US (e.g., "CA")
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
    
    elif backend == DetectorBackend.SPECIESNET:
        return SpeciesNetDetector(
            model_version=kwargs.get("model_version", "v4.0.2a"),
            country=kwargs.get("country"),
            admin1_region=kwargs.get("admin1_region"),
            cache_dir=kwargs.get("cache_dir"),
        )
    
    else:
        raise ValueError(f"Unknown detector backend: {backend}")


# Keep backward compatibility
__all__ = [
    "Detection",
    "DetectorBackend", 
    "BaseDetector",
    "YoloDetector",
    "SpeciesNetDetector",
    "create_detector",
]
