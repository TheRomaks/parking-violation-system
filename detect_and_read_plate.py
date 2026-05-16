from parking_vision.common.video import open_writer, resolve_source
from parking_vision.plate_reading import (
    ALLOWLIST,
    CONF_THRESHOLD,
    EasyOCRReader,
    MAX_AR,
    MIN_AR,
    MIN_AREA,
    MODEL_PATH,
    PaddleOCRReader,
    PlateReader,
    RUS_PLATE_PATTERN,
    VideoOutputConfig,
    crop_bbox,
    is_valid_plate,
    main,
    normalize_plate,
    parse_args,
    preprocess,
    run_plate_reading,
)
from parking_vision.plate_reading.io import save_detections_csv, save_detections_jsonl

__all__ = [
    "ALLOWLIST",
    "CONF_THRESHOLD",
    "EasyOCRReader",
    "MAX_AR",
    "MIN_AR",
    "MIN_AREA",
    "MODEL_PATH",
    "PaddleOCRReader",
    "PlateReader",
    "RUS_PLATE_PATTERN",
    "VideoOutputConfig",
    "crop_bbox",
    "is_valid_plate",
    "main",
    "normalize_plate",
    "open_writer",
    "parse_args",
    "preprocess",
    "resolve_source",
    "run_plate_reading",
    "save_detections_csv",
    "save_detections_jsonl",
]


if __name__ == "__main__":
    main()
