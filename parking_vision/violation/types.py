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
    zone_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    side_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    hard_road_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    zone_distance_transform: np.ndarray | None = field(default=None, repr=False, compare=False)
    _bbox_x1: int = field(default=0, repr=False)
    _bbox_y1: int = field(default=0, repr=False)
    _bbox_x2: int = field(default=0, repr=False)
    _bbox_y2: int = field(default=0, repr=False)

    def contains_point(self, x: float, y: float) -> bool:
        if not (self._bbox_x1 <= x <= self._bbox_x2 and self._bbox_y1 <= y <= self._bbox_y2):
            return False
        if self.zone_mask is not None:
            ix = int(round(x))
            iy = int(round(y))
            h, w = self.zone_mask.shape[:2]
            return 0 <= ix < w and 0 <= iy < h and self.zone_mask[iy, ix] > 0
        return cv2.pointPolygonTest(np.array(self.polygon, dtype=np.int32), (x, y), False) >= 0

    def bbox_tuple(self) -> tuple[int, int, int, int]:
        return self._bbox_x1, self._bbox_y1, self._bbox_x2, self._bbox_y2

    def to_dict(self) -> dict[str, Any]:
        return {
            "sign_id": self.sign_id,
            "sign_label": self.sign_label,
            "polygon": list(self.polygon),
            "source_bbox": self.source_bbox.to_dict(),
            "time_limit_s": self.time_limit_s,
            "restriction": self.restriction,
            "applies_now": self.applies_now,
            "side": self.side,
            "direction": self.direction,
            "plate_labels": list(self.plate_labels),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ZoneAssignment:
    track_id: int
    zone: SignZone
    probability: float
    decision: str
    reasons: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def applies(self) -> bool:
        return self.decision == "applies"

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "sign_id": self.zone.sign_id,
            "sign_label": self.zone.sign_label,
            "probability": float(self.probability),
            "decision": self.decision,
            "reasons": dict(self.reasons),
            "metadata": dict(self.metadata),
        }


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
    zone_assignments: list[ZoneAssignment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "car_detections": [item.to_dict() for item in self.car_detections],
            "sign_detections": [item.to_dict() for item in self.sign_detections],
            "plate_matches": self.plate_matches,
            "active_zones": [zone.to_dict() for zone in self.active_zones],
            "active_violations": [violation.to_dict() for violation in self.active_violations],
            "zone_assignments": [assignment.to_dict() for assignment in self.zone_assignments],
        }


@dataclass(slots=True)
class VideoOutputConfig:
    annotated_video_path: Path
    violations_csv_path: Path
    jsonl_path: Path