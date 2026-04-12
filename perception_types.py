import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class Detection:
    module: str
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    track_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = self.bbox.to_dict()
        return payload

    def to_csv_row(
        self,
        frame_index: int,
        timestamp_ms: float | None,
    ) -> list[Any]:
        return [
            frame_index,
            "" if timestamp_ms is None else f"{timestamp_ms:.2f}",
            self.module,
            self.track_id if self.track_id is not None else "",
            self.class_id,
            self.class_name,
            f"{self.confidence:.4f}",
            int(self.bbox.x1),
            int(self.bbox.y1),
            int(self.bbox.x2),
            int(self.bbox.y2),
            json.dumps(self.metadata, ensure_ascii=False),
        ]


@dataclass(slots=True)
class FrameDetections:
    frame_index: int
    timestamp_ms: float | None
    detections: list[Detection]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "detections": [item.to_dict() for item in self.detections],
        }


CSV_HEADER = [
    "frame_index",
    "timestamp_ms",
    "module",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "metadata",
]


def ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def append_frame_to_csv(csv_writer: csv.writer, frame_result: FrameDetections) -> None:
    for detection in frame_result.detections:
        csv_writer.writerow(
            detection.to_csv_row(
                frame_index=frame_result.frame_index,
                timestamp_ms=frame_result.timestamp_ms,
            )
        )


def save_frame_results_jsonl(output_path: Path, frame_results: list[FrameDetections]) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        for frame_result in frame_results:
            file.write(json.dumps(frame_result.to_dict(), ensure_ascii=False) + "\n")
