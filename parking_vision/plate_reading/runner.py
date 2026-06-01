from dataclasses import dataclass
from pathlib import Path

import cv2

from parking_vision.common.video import open_writer

from .io import save_detections_csv, save_detections_jsonl
from .reader import PlateReader


@dataclass(slots=True)
class VideoOutputConfig:
    annotated_video_path: Path
    csv_path: Path
    jsonl_path: Path


def run_plate_reading(
    source: int | str,
    reader: PlateReader,
    output_config: VideoOutputConfig,
    show: bool = False,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read first frame")

    writer = open_writer(output_config.annotated_video_path, cap, first_frame)
    frame_index = 0
    all_detections: list[list[dict[str, object]]] = []

    try:
        while True:
            if frame_index == 0:
                frame = first_frame
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            plates = reader.process_frame(frame)
            all_detections.append(plates)

            for plate in plates:
                x1, y1, x2, y2 = map(int, plate["bbox"])
                color = (0, 255, 0) if plate["valid"] else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = plate["text"] or "plate"
                label = f"{label_text} ({plate['ocr_conf']:.2f})"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            writer.write(frame)
            if show:
                cv2.imshow("Plate Reader", frame)
                if cv2.waitKey(1) == 27:
                    break

            frame_index += 1
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    save_detections_csv(all_detections, output_config.csv_path)
    save_detections_jsonl(all_detections, output_config.jsonl_path)
