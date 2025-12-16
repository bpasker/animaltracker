"""Detection helpers and engine adapters.

This file provides adapters for supported inference engines. The original
YoloDetector remains for backward compatibility; CameraTrapAIDetector is a
best-effort adapter for the CameraTrapAI Python package. A `create_detector`
factory returns an instance by engine name.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Protocol, Any

import numpy as np

# Optional import for the Ultralytics YOLO package.
try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class Detection:
    species: str
    confidence: float
    bbox: List[float]


class DetectorProtocol(Protocol):
    def infer(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        ...


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


class CameraTrapAIDetector:
    """
    Minimal CameraTrapAI adapter.
    The upstream Python API is evolving; this adapter attempts to support
    a few common patterns and normalize outputs to `Detection`.
    """

    def __init__(self, model_path: Optional[str] = None, class_map: Optional[Dict[int, str]] = None) -> None:
        # Import package lazily to avoid a hard dependency in all installs
        ct_pkg = None
        for name in ("cameratrapai", "camera_trap_ai", "camera_trap_ai_predict"):
            try:
                ct_pkg = __import__(name)
                break
            except Exception:
                ct_pkg = None

        if ct_pkg is None:
            raise RuntimeError("CameraTrapAI package not installed - run pip install cameratrapai")

        self._pkg = ct_pkg
        self.model_path = model_path
        self.class_map = class_map or {}
        # Attempt to load model-like objects if available
        self.model = None
        if hasattr(ct_pkg, "InferenceModel"):
            cls = getattr(ct_pkg, "InferenceModel")
            try:
                self.model = cls.load(model_path) if model_path else cls()
            except Exception:
                try:
                    self.model = cls(model_path) if model_path else cls()
                except Exception:
                    self.model = None
        elif hasattr(ct_pkg, "Model"):
            cls = getattr(ct_pkg, "Model")
            try:
                self.model = cls.load(model_path) if model_path else cls()
            except Exception:
                try:
                    self.model = cls(model_path) if model_path else cls()
                except Exception:
                    self.model = None
        elif hasattr(ct_pkg, "load_model"):
            try:
                self.model = ct_pkg.load_model(model_path)
            except Exception:
                self.model = None

        if self.model is None:
            LOGGER.info("CameraTrapAI: loaded package but no model object found. Falling back to package-level predict functions")

    def _normalize_item(self, item: Any) -> Optional[Detection]:
        # Support dict-like responses
        if isinstance(item, dict):
            bbox = item.get("bbox") or item.get("boxes") or item.get("box")
            score = item.get("score") or item.get("confidence") or item.get("conf")
            label = item.get("label") or item.get("species") or item.get("class")
            if bbox is None or score is None or label is None:
                return None
            try:
                bbox_list = [float(x) for x in bbox]
                return Detection(species=str(label), confidence=float(score), bbox=bbox_list)
            except Exception:
                return None

        # Try simple object-based APIs
        try:
            boxes = getattr(item, "boxes", None) or getattr(item, "bboxes", None)
            scores = getattr(item, "scores", None) or getattr(item, "confs", None)
            labels = getattr(item, "labels", None) or getattr(item, "classes", None)
            if boxes is None:
                return None
            # single item assume first detection only
            if len(boxes) and hasattr(boxes[0], "__len__"):
                b = boxes[0]
                if isinstance(b, (list, tuple)) or hasattr(b, "__len__"):
                    bbox_list = [float(x) for x in b]
                    conf = float(scores[0]) if scores is not None else 1.0
                    label = labels[0] if labels is not None else "unknown"
                    return Detection(species=str(label), confidence=conf, bbox=bbox_list)
        except Exception:
            return None
        return None

    def infer(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        detections: List[Detection] = []

        # If we have a loaded model with a predict() method
        if self.model is not None:
            predict_fn = getattr(self.model, "predict", None) or getattr(self.model, "inference", None)
            if predict_fn is None:
                raise RuntimeError("Loaded CameraTrapAI model does not expose 'predict' or 'inference' method")
            raw = predict_fn(frame)
            if isinstance(raw, list):
                for item in raw:
                    det = self._normalize_item(item)
                    if det and det.confidence >= conf_threshold:
                        detections.append(det)
                return detections
            det = self._normalize_item(raw)
            return [det] if det and det.confidence >= conf_threshold else []

        # Fallback to package-level functions
        for fname in ("predict_image", "inference", "detect_image", "predict"):
            if hasattr(self._pkg, fname):
                fn = getattr(self._pkg, fname)
                try:
                    raw = fn(frame, model_path=self.model_path)
                except TypeError:
                    raw = fn(frame)
                if isinstance(raw, list):
                    for item in raw:
                        det = self._normalize_item(item)
                        if det and det.confidence >= conf_threshold:
                            detections.append(det)
                    return detections
                det = self._normalize_item(raw)
                return [det] if det and det.confidence >= conf_threshold else []

        raise RuntimeError("CameraTrapAI installed but no compatible inference entrypoint found. Adapt CameraTrapAIDetector to your package.")


def create_detector(engine: str = "cameratrapai", model_path: Optional[str] = None) -> DetectorProtocol:
    engine = (engine or "cameratrapai").lower()
    if engine in ("ctai", "cameratrapai", "camera_trap_ai"):
        return CameraTrapAIDetector(model_path=model_path)
    if engine in ("yolo", "ultralytics"):
        return YoloDetector(model_path=model_path)
    raise ValueError(f"Unknown engine: {engine}")
