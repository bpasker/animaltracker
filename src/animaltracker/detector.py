"""YOLO detection helpers."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class Detection:
    species: str
    confidence: float
    bbox: List[float]


class YoloDetector:
    def __init__(self, model_path: str = "yolov8n.pt", class_map: dict[int, str] | None = None) -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not installed - run pip install ultralytics")
        self.model = YOLO(model_path)
        self.class_map = class_map or self.model.names

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
                detections.append(Detection(species=species, confidence=conf, bbox=bbox))
        return detections
