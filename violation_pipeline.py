from parking_vision.common.video import open_writer, resolve_source
from parking_vision.violation import (
    PipelineFrameResult,
    SignZone,
    VideoOutputConfig,
    ViolationPipeline,
    ViolationRecord,
    main,
    parse_args,
    run_pipeline,
)
from parking_vision.violation.constants import (
    END_SIGN_ID,
    PARKING_SIGN_IDS,
    PROHIBITORY_SIGN_IDS,
    SIGN_LABELS,
    SIGN_TIME_LIMITS_S,
)
from parking_vision.violation.geometry import bbox_intersection_area as _bbox_intersection_area
from parking_vision.violation.geometry import bbox_tuple_iou as _bbox_tuple_iou
from parking_vision.violation.geometry import polygon_bbox as _poly_bbox
from parking_vision.violation.io import save_jsonl
from parking_vision.violation.zone_manager import SignZoneManager
from parking_vision.violation.car_state import CarStateManager

__all__ = [
    "CarStateManager",
    "END_SIGN_ID",
    "PARKING_SIGN_IDS",
    "PROHIBITORY_SIGN_IDS",
    "PipelineFrameResult",
    "SIGN_LABELS",
    "SIGN_TIME_LIMITS_S",
    "SignZone",
    "SignZoneManager",
    "VideoOutputConfig",
    "ViolationPipeline",
    "ViolationRecord",
    "_bbox_intersection_area",
    "_bbox_tuple_iou",
    "_poly_bbox",
    "main",
    "open_writer",
    "parse_args",
    "resolve_source",
    "run_pipeline",
    "save_jsonl",
]


if __name__ == "__main__":
    main()
