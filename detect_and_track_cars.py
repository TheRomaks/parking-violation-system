import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from perception_types import (
    CSV_HEADER,
    BoundingBox,
    Detection,
    FrameDetections,
    append_frame_to_csv,
    ensure_parent_dir,
    save_frame_results_jsonl,
)


MODULE_NAME = "car_tracker"


@dataclass(slots=True)
class VideoOutputConfig:
    annotated_video_path: Path
    csv_path: Path
    jsonl_path: Path


class CarTracker:
    def __init__(
        self,
        model_path: str = "models/cars.pt",
        tracker_config: str = "bytetrack.yaml",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1280,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.tracker_config = tracker_config
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def process_frame(
        self,
        frame: Any,
        frame_index: int,
        timestamp_ms: float | None = None,
    ) -> FrameDetections:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        detections: list[Detection] = []

        if boxes is not None and len(boxes) > 0:
            xyxy_list = boxes.xyxy.cpu().tolist()
            conf_list = boxes.conf.cpu().tolist()
            cls_list = boxes.cls.cpu().tolist()
            id_list = (
                boxes.id.int().cpu().tolist()
                if boxes.id is not None
                else [None] * len(xyxy_list)
            )

            for box, conf, cls_id, track_id in zip(
                xyxy_list,
                conf_list,
                cls_list,
                id_list,
            ):
                class_id = int(cls_id)
                class_name = result.names.get(class_id, str(class_id))
                bbox = BoundingBox(*box)
                detections.append(
                    Detection(
                        module=MODULE_NAME,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(conf),
                        bbox=bbox,
                        track_id=None if track_id is None else int(track_id),
                        metadata={
                            "model": self.model_path.name,
                            "tracker": self.tracker_config,
                        },
                    )
                )

        return FrameDetections(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            detections=detections,
        )

    @staticmethod
    def annotate_frame(frame: Any, frame_result: FrameDetections) -> Any:
        annotated = frame.copy()
        for detection in frame_result.detections:
            x1, y1, x2, y2 = detection.bbox.to_int_tuple()
            color = (0, 200, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            track_label = detection.track_id if detection.track_id is not None else "NA"
            label = (
                f"{detection.class_name} "
                f"ID:{track_label} "
                f"conf:{detection.confidence:.2f}"
            )
            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )
            text_y = max(y1 - 10, text_h + 6)
            cv2.rectangle(
                annotated,
                (x1, text_y - text_h - 6),
                (x1 + text_w + 8, text_y),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, text_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
        return annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Car detection and tracking module for the parking violation system."
    )
    parser.add_argument("--source", required=True, help="Video path or camera index.")
    parser.add_argument("--model", default="models/cars.pt", help="Car model weights.")
    parser.add_argument(
        "--output",
        default="outputs/tracked_cars.mp4",
        help="Path to annotated output video.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default="outputs/tracked_cars.csv",
        help="Flat CSV export for downstream modules.",
    )
    parser.add_argument(
        "--jsonl",
        dest="jsonl_path",
        default="outputs/tracked_cars.jsonl",
        help="Frame-by-frame JSONL export for downstream modules.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display frames during processing.",
    )
    return parser.parse_args()


def resolve_source(source_value: str) -> int | str:
    return int(source_value) if source_value.isdigit() else source_value


def open_writer(output_path: Path, capture: cv2.VideoCapture, frame: Any) -> cv2.VideoWriter:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def run_tracking(
    source: int | str,
    tracker: CarTracker,
    output_config: VideoOutputConfig,
    show: bool = False,
) -> list[FrameDetections]:
    ensure_parent_dir(output_config.annotated_video_path)
    ensure_parent_dir(output_config.csv_path)
    ensure_parent_dir(output_config.jsonl_path)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"Cannot read the first frame from source: {source}")

    writer = open_writer(output_config.annotated_video_path, capture, first_frame)
    frame_results: list[FrameDetections] = []

    try:
        with output_config.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(CSV_HEADER)

            frame_index = 0
            while True:
                frame = first_frame if frame_index == 0 else None
                if frame is None:
                    ok, frame = capture.read()
                    if not ok:
                        break

                timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_value = None if timestamp_ms < 0 else float(timestamp_ms)
                frame_result = tracker.process_frame(
                    frame=frame,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_value,
                )
                frame_results.append(frame_result)
                append_frame_to_csv(csv_writer, frame_result)

                annotated_frame = tracker.annotate_frame(frame, frame_result)
                writer.write(annotated_frame)

                if show:
                    cv2.imshow("Car Tracking", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                frame_index += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    save_frame_results_jsonl(output_config.jsonl_path, frame_results)
    return frame_results


def main() -> None:
    args = parse_args()
    tracker = CarTracker(
        model_path=args.model,
        tracker_config=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    output_config = VideoOutputConfig(
        annotated_video_path=Path(args.output),
        csv_path=Path(args.csv_path),
        jsonl_path=Path(args.jsonl_path),
    )
    run_tracking(
        source=resolve_source(args.source),
        tracker=tracker,
        output_config=output_config,
        show=args.show,
    )
    print(f"Annotated video saved to: {output_config.annotated_video_path}")
    print(f"CSV export saved to: {output_config.csv_path}")
    print(f"JSONL export saved to: {output_config.jsonl_path}")


if __name__ == "__main__":
    main()
