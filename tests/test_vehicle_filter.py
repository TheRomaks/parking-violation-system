import numpy as np

from perception_types import BoundingBox, Detection
from parking_vision.violation.vehicle_filter import VehicleCandidateFilter


def _car(track_id: int, bbox: tuple[float, float, float, float], confidence: float = 0.8) -> Detection:
    return Detection(
        module="car",
        class_id=0,
        class_name="car",
        confidence=confidence,
        bbox=BoundingBox(*bbox),
        track_id=track_id,
    )


def test_vehicle_filter_accepts_vehicle_supported_by_road_mask() -> None:
    road = np.zeros((100, 120), dtype=np.uint8)
    road[55:100, 10:110] = 255
    vehicle_filter = VehicleCandidateFilter()

    result = vehicle_filter.filter([_car(1, (30, 35, 80, 95))], road.shape, road, [])

    assert [item.track_id for item in result] == [1]
    assert result[0].metadata["candidate_filter"] == "accepted"


def test_vehicle_filter_rejects_vehicle_without_road_or_zone_support() -> None:
    road = np.zeros((100, 120), dtype=np.uint8)
    road[70:100, 10:110] = 255
    vehicle_filter = VehicleCandidateFilter()
    candidate = _car(1, (30, 10, 80, 55))

    result = vehicle_filter.filter([candidate], road.shape, road, [])

    assert result == []
    assert candidate.metadata["candidate_filter"] == "rejected"


def test_vehicle_filter_suppresses_nested_occluded_boxes() -> None:
    road = np.zeros((120, 160), dtype=np.uint8)
    road[40:120, 5:155] = 255
    vehicle_filter = VehicleCandidateFilter()
    large = _car(1, (20, 35, 130, 115), confidence=0.9)
    nested = _car(2, (40, 50, 95, 105), confidence=0.4)

    result = vehicle_filter.filter([large, nested], road.shape, road, [])

    assert [item.track_id for item in result] == [1]
    assert nested.metadata["candidate_filter"] == "occluded"
