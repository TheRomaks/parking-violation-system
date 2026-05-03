import csv
from dataclasses import dataclass
from pathlib import Path

import cv2

from parking_vision.common.video import open_writer
from perception_types import (
    CSV_HEADER,
    FrameDetections,
    append_frame_to_csv,
    ensure_parent_dir,
    save_frame_results_jsonl,
)

from .detector import SignDetector


@dataclass(slots=True)
class VideoOutputConfig:
    annotated_video_path: Path
    csv_path: Path
    jsonl_path: Path


def run_detection(
    source: int | str,
    detector: SignDetector,
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
                frame_result = detector.process_frame(
                    frame=frame,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_value,
                )
                frame_results.append(frame_result)
                append_frame_to_csv(csv_writer, frame_result)

                annotated_frame = detector.annotate_frame(frame, frame_result)
                writer.write(annotated_frame)

                if show:
                    cv2.imshow("Sign Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                frame_index += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    save_frame_results_jsonl(output_config.jsonl_path, frame_results)
    return frame_results
