from typing import Any

import numpy as np
from ultralytics import YOLO

from .constants import (
    ALLOWLIST,
    CONF_THRESHOLD,
    MAX_AR,
    MIN_AR,
    MIN_AREA,
    MODEL_PATH,
    RUS_PLATE_PATTERN,
)
from .ocr import EasyOCRReader
from .preprocessing import crop_bbox, is_valid_plate, normalize_plate, preprocess


class PlateReader:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        conf: float = 0.3,
        iou: float = 0.5,
        imgsz: int = 1280,
    ) -> None:
        self.model = YOLO(model_path)
        self.ocr = EasyOCRReader()
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def process_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self.model(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz)[0]
        plates: list[dict[str, Any]] = []

        if results.boxes is None:
            return plates

        for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
            bbox = box.cpu().tolist()
            detection_conf = float(conf)
            if not is_valid_plate(bbox, detection_conf):
                continue

            crop = crop_bbox(frame, bbox)
            prepared_crop = preprocess(crop)
            if prepared_crop is None:
                continue

            text, score = self.ocr.read(prepared_crop)
            if score < 0.4:
                continue

            plates.append(
                {
                    "bbox": bbox,
                    "conf": detection_conf,
                    "text": text,
                    "ocr_conf": score,
                    "valid": bool(RUS_PLATE_PATTERN.match(text)),
                }
            )

        return plates


__all__ = [
    "ALLOWLIST",
    "CONF_THRESHOLD",
    "EasyOCRReader",
    "MAX_AR",
    "MIN_AR",
    "MIN_AREA",
    "MODEL_PATH",
    "PlateReader",
    "RUS_PLATE_PATTERN",
    "crop_bbox",
    "is_valid_plate",
    "normalize_plate",
    "preprocess",
]
