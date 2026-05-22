from perception_types import BoundingBox, Detection

from parking_vision.violation.types import SignZone
from parking_vision.violation.zone_reasoner import ZoneReasoner


def test_zone_reasoner_accepts_high_confidence_membership_immediately() -> None:
    zone = SignZone(
        sign_id=0,
        sign_label="3.27",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        source_bbox=BoundingBox(0, 0, 10, 10),
        time_limit_s=0.0,
        metadata={
            "zone_confidence": 0.8,
            "road_mask_support": 0.8,
            "zone_start_point": [0.0, 0.0],
            "zone_direction_vec": [1.0, 0.0],
        },
        _bbox_x1=0,
        _bbox_y1=0,
        _bbox_x2=100,
        _bbox_y2=100,
    )
    car = Detection(
        module="car",
        class_id=0,
        class_name="car",
        confidence=0.9,
        bbox=BoundingBox(30, 40, 70, 90),
        track_id=1,
    )

    reasoner = ZoneReasoner()
    assignment = reasoner.assign_car_to_zone(car, [zone], timestamp_ms=0.0)

    assert assignment is not None
    assert assignment.decision == "applies"
    assert assignment.probability >= reasoner.apply_threshold


def test_zone_reasoner_rejects_vehicle_before_sign_start() -> None:
    zone = SignZone(
        sign_id=0,
        sign_label="3.27",
        polygon=[(-50, 0), (150, 0), (150, 100), (-50, 100)],
        source_bbox=BoundingBox(45, 0, 55, 20),
        time_limit_s=0.0,
        metadata={
            "zone_confidence": 0.9,
            "road_mask_support": 0.9,
            "zone_start_point": [50.0, 0.0],
            "zone_direction_vec": [1.0, 0.0],
        },
        _bbox_x1=-50,
        _bbox_y1=0,
        _bbox_x2=150,
        _bbox_y2=100,
    )
    car = Detection(
        module="car",
        class_id=0,
        class_name="car",
        confidence=0.9,
        bbox=BoundingBox(-20, 40, 20, 90),
        track_id=1,
    )

    reasoner = ZoneReasoner()
    first = reasoner.assign_car_to_zone(car, [zone], timestamp_ms=0.0)
    second = reasoner.assign_car_to_zone(car, [zone], timestamp_ms=40.0)

    assert first is not None
    assert second is not None
    assert first.decision == "not_applies"
    assert second.decision == "not_applies"
    assert first.reasons["start_boundary"] < 0.0
    assert first.metadata["start_relation"] == "before_start"


def test_zone_reasoner_uses_start_boundary_instead_of_image_order() -> None:
    zone = SignZone(
        sign_id=0,
        sign_label="3.27",
        polygon=[(300, 200), (240, 520), (520, 560), (560, 240)],
        source_bbox=BoundingBox(340, 180, 370, 230),
        time_limit_s=0.0,
        side="right",
        direction="forward",
        metadata={
            "zone_confidence": 0.9,
            "road_mask_support": 0.9,
            "zone_start_point": [300.0, 200.0],
            "zone_direction_vec": [-0.18, 0.98],
        },
        _bbox_x1=240,
        _bbox_y1=200,
        _bbox_x2=560,
        _bbox_y2=560,
    )
    car = Detection(
        module="car",
        class_id=0,
        class_name="car",
        confidence=0.9,
        bbox=BoundingBox(380, 360, 500, 520),
        track_id=1,
    )

    reasoner = ZoneReasoner()
    assignment = reasoner.assign_car_to_zone(car, [zone], timestamp_ms=0.0)

    assert assignment is not None
    assert assignment.decision == "applies"
    assert "image_order_start" not in assignment.reasons
    assert "image_order_relation" not in assignment.metadata
    assert assignment.metadata["start_relation"] == "after_start"
