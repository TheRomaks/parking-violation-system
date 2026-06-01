from perception_types import BoundingBox, Detection

from parking_vision.violation.car_state import CarStateManager
from parking_vision.violation.pipeline import ViolationPipeline


def _car(track_id: int = 1) -> Detection:
    return Detection(
        module="car",
        class_id=0,
        class_name="car",
        confidence=0.9,
        bbox=BoundingBox(20, 20, 180, 120),
        track_id=track_id,
    )


def test_plate_candidate_selection_prefers_valid_plate_for_track() -> None:
    manager = CarStateManager()
    manager.update_track(_car(), timestamp_ms=0.0, assignment=None, plate_text="")

    manager.update_plate_candidate(
        track_id=1,
        text="P75BBA178",
        ocr_conf=0.88,
        detection_conf=0.8,
        valid=False,
        bbox=[40, 70, 120, 90],
        timestamp_ms=40.0,
    )
    state = manager.update_plate_candidate(
        track_id=1,
        text="P576BA178",
        ocr_conf=0.62,
        detection_conf=0.8,
        valid=True,
        bbox=[42, 71, 121, 91],
        timestamp_ms=80.0,
    )

    assert state["plate"] == "P576BA178"


def test_late_clear_plate_can_replace_repeated_older_candidate() -> None:
    manager = CarStateManager()
    manager.update_track(_car(), timestamp_ms=0.0, assignment=None, plate_text="")

    for index in range(6):
        manager.update_plate_candidate(
            track_id=1,
            text="P576BA178",
            ocr_conf=0.62,
            detection_conf=0.78,
            valid=True,
            bbox=[42, 71, 121, 91],
            timestamp_ms=float(index * 160),
        )

    state = manager.update_plate_candidate(
        track_id=1,
        text="O576BO178",
        ocr_conf=0.85,
        detection_conf=0.86,
        valid=True,
        bbox=[250, 420, 355, 448],
        timestamp_ms=5000.0,
    )

    assert state["plate"] == "O576BO178"


def test_unreadable_plate_detection_keeps_bbox_without_clearing_best_text() -> None:
    manager = CarStateManager()
    manager.update_track(_car(), timestamp_ms=0.0, assignment=None, plate_text="")
    manager.update_plate_candidate(1, "P576BA178", ocr_conf=0.7, detection_conf=0.8, valid=True)

    state = manager.update_plate_candidate(
        track_id=1,
        text="",
        ocr_conf=0.0,
        detection_conf=0.9,
        valid=False,
        bbox=[50, 75, 130, 95],
        timestamp_ms=120.0,
    )

    assert state["plate"] == "P576BA178"
    assert state["plate_bbox"] == [50.0, 75.0, 130.0, 95.0]


def test_plate_matching_keeps_unreadable_plate_observation() -> None:
    pipeline = ViolationPipeline.__new__(ViolationPipeline)
    car = _car()
    plate_result = [
        {
            "bbox": [60, 78, 125, 96],
            "conf": 0.83,
            "text": "",
            "ocr_conf": 0.0,
            "valid": False,
            "readable": False,
        }
    ]

    matches = pipeline._match_plates_to_cars([car], plate_result)

    assert 1 in matches
    assert matches[1]["bbox"] == [60, 78, 125, 96]
    assert matches[1]["text"] == ""
