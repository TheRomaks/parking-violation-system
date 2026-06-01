import csv

import cv2

from parking_vision.common.video import open_writer
from perception_types import ensure_parent_dir

from .io import save_jsonl
from .pipeline import ViolationPipeline
from .types import PipelineFrameResult, VideoOutputConfig


def run_pipeline(
    source: int | str,
    pipeline: ViolationPipeline,
    out: VideoOutputConfig,
    show: bool = False,
) -> list[PipelineFrameResult]:
    ensure_parent_dir(out.annotated_video_path)
    ensure_parent_dir(out.violations_csv_path)
    ensure_parent_dir(out.jsonl_path)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("Cannot read first frame")

    writer = open_writer(out.annotated_video_path, capture, first_frame)
    results: list[PipelineFrameResult] = []
    emitted_violation_keys: set[tuple[int, int]] = set()

    try:
        with out.violations_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer_csv = csv.writer(csv_file)
            writer_csv.writerow(
                [
                    "frame_index",
                    "timestamp_ms",
                    "track_id",
                    "plate",
                    "sign_id",
                    "sign_label",
                    "status",
                    "time_in_zone_s",
                    "stopped_duration_s",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ]
            )

            frame_index = 0
            while True:
                frame = first_frame if frame_index == 0 else None
                if frame is None:
                    ok, frame = capture.read()
                    if not ok:
                        break

                timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_value = None if timestamp_ms < 0 else float(timestamp_ms)

                annotated_frame, violations, result = pipeline.process_frame(
                    frame,
                    frame_index,
                    timestamp_value,
                )
                results.append(result)
                writer.write(annotated_frame)

                for violation in violations:
                    violation_key = (violation.track_id, violation.sign_id)
                    if violation_key in emitted_violation_keys:
                        continue
                    emitted_violation_keys.add(violation_key)
                    writer_csv.writerow(
                        [
                            frame_index,
                            "" if timestamp_value is None else f"{timestamp_value:.2f}",
                            violation.track_id,
                            violation.plate,
                            violation.sign_id,
                            violation.sign_label,
                            violation.status,
                            f"{violation.time_in_zone_s:.2f}",
                            f"{violation.stopped_duration_s:.2f}",
                            int(violation.bbox.x1),
                            int(violation.bbox.y1),
                            int(violation.bbox.x2),
                            int(violation.bbox.y2),
                        ]
                    )

                if show:
                    cv2.imshow("Violation Pipeline", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                frame_index += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    save_jsonl(out.jsonl_path, results)
    return results
