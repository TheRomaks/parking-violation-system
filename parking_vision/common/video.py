from pathlib import Path
from typing import Any

import cv2


def resolve_source(source_value: str) -> int | str:
    return int(source_value) if source_value.isdigit() else source_value


def open_writer(
    output_path: Path,
    capture: cv2.VideoCapture,
    frame: Any,
) -> cv2.VideoWriter:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
