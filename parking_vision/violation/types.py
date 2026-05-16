from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from perception_types import BoundingBox, Detection


@dataclass(slots=True)
class SignZone:
    sign_id: int
    sign_label: str
    polygon: list[tuple[int, int]]
    source_bbox: BoundingBox
    time_limit_s: float
    restriction: str = "prohibition"
    applies_now: bool = True
    side: str = "unknown"
    direction: str = "forward"
    plate_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _bbox_x1: int = field(default=0, repr=False)
    _bbox_y1: int = field(default=0, repr=False)
    _bbox_x2: int = field(default=0, repr=False)
    _bbox_y2: int = field(default=0, repr=False)

    def contains_point(self, x: float, y: float) -> bool:
        if not (self._bbox_x1 <= x <= self._bbox_x2 and self._bbox_y1 <= y <= self._bbox_y2):
            return False
        return cv2.pointPolygonTest(np.array(self.polygon, dtype=np.int32), (x, y), False) >= 0

    def bbox_tuple(self) -> tuple[int, int, int, int]:
        return self._bbox_x1, self._bbox_y1, self._bbox_x2, self._bbox_y2

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_bbox"] = self.source_bbox.to_dict()
        for key in ("_bbox_x1", "_bbox_y1", "_bbox_x2", "_bbox_y2"):
            payload.pop(key, None)
        return payload


@dataclass(slots=True)
class ViolationRecord:
    track_id: int
    plate: str
    sign_id: int
    sign_label: str
    status: str
    time_in_zone_s: float
    stopped_duration_s: float
    bbox: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = self.bbox.to_dict()
        return payload


@dataclass(slots=True)
class PipelineFrameResult:
    frame_index: int
    timestamp_ms: float | None
    car_detections: list[Detection]
    sign_detections: list[Detection]
    plate_matches: dict[int, str]
    active_zones: list[SignZone]
    active_violations: list[ViolationRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "car_detections": [item.to_dict() for item in self.car_detections],
            "sign_detections": [item.to_dict() for item in self.sign_detections],
            "plate_matches": self.plate_matches,
            "active_zones": [zone.to_dict() for zone in self.active_zones],
            "active_violations": [violation.to_dict() for violation in self.active_violations],
        }


@dataclass(slots=True)
class VideoOutputConfig:
    annotated_video_path: Path
    violations_csv_path: Path
    jsonl_path: Path
