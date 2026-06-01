import csv
import json
from pathlib import Path
from typing import Any


def save_detections_csv(plates_list: list[list[dict[str, Any]]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames: list[str] = []
        for frame_plates in plates_list:
            for plate in frame_plates:
                for key in plate.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
        if fieldnames:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for frame_plates in plates_list:
                for plate in frame_plates:
                    writer.writerow(plate)


def save_detections_jsonl(plates_list: list[list[dict[str, Any]]], jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as file:
        for frame_idx, frame_plates in enumerate(plates_list):
            frame_data = {
                "frame_index": frame_idx,
                "plates": frame_plates,
            }
            file.write(json.dumps(frame_data, ensure_ascii=False) + "\n")
