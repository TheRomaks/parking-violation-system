import numpy as np

from perception_types import BoundingBox
from parking_vision.violation.sign_rules import SignRule
from parking_vision.violation.zone_manager import SignZoneManager


def test_action_mask_is_road_subset_and_starts_at_sign_projection() -> None:
    manager = SignZoneManager.__new__(SignZoneManager)
    road_mask = np.zeros((80, 120), dtype=np.uint8)
    road_mask[20:70, 20:100] = 255

    zone_mask = manager._select_split_components(
        side_mask=road_mask,
        anchor_point=np.array([50.0, 20.0], dtype=np.float32),
        split_line_dir=np.array([0.0, 1.0], dtype=np.float32),
        direction=np.array([1.0, 0.0], dtype=np.float32),
        mode="forward",
        frame_w=120,
        frame_h=80,
        length_px=40.0,
    )

    assert zone_mask is not None
    assert np.count_nonzero((zone_mask > 0) & (road_mask == 0)) == 0
    assert np.count_nonzero(zone_mask[:, :48]) == 0
    assert np.count_nonzero(zone_mask[:, 52:88]) > 0


def test_sign_anchor_projects_down_to_road_edge_not_bbox_bottom() -> None:
    manager = SignZoneManager.__new__(SignZoneManager)
    manager._anchor_search_margin_px = 170

    detection = type(
        "DetectionStub",
        (),
        {
            "bbox": type(
                "BBoxStub",
                (),
                {"to_int_tuple": lambda self: (92, 70, 112, 110)},
            )()
        },
    )()
    profile = [(y, 20, 82 + int((y - 120) * 0.05)) for y in range(120, 281, 8)]

    anchor = manager._estimate_anchor_on_road_edge(
        detection=detection,
        profile=profile,
        frame_h=320,
        locked_side="right",
    )

    assert anchor is not None
    anchor_y, side, _ = anchor
    assert side == "right"
    assert anchor_y > 170


def test_right_side_default_zone_direction_points_after_sign() -> None:
    manager = SignZoneManager.__new__(SignZoneManager)

    direction = manager._preferred_rule_direction(
        tangent=np.array([0.7, -0.7], dtype=np.float32),
        side="right",
        rule=None,
    )

    assert direction[0] < 0.0


def test_weak_visible_sign_zone_stays_active_without_full_lock() -> None:
    manager = SignZoneManager.__new__(SignZoneManager)
    manager._max_missing_frames = 30
    manager._sign_presence_confidence = 0.25

    observed_ids: set[int] = {1}
    manager._states = {
        1: {
            "locked": False,
            "missed_frames": 0,
            "polygon": [(20, 20), (80, 20), (80, 70), (20, 70)],
            "rule": SignRule(
                sign_id=0,
                sign_label="3.27",
                restriction="no_stopping",
                time_limit_s=0.0,
                applies_now=True,
            ),
            "source_bbox": BoundingBox(90, 10, 110, 40),
            "sign_label": "3.27",
            "side": "right",
            "direction_vec": np.array([1.0, 0.0], dtype=np.float32),
            "zone_mask": np.zeros((100, 120), dtype=np.uint8),
            "side_mask": None,
            "zone_metadata": {},
            "last_confidence": 0.72,
            "current_confidence": 0.28,
        },
        2: {
            "locked": False,
            "missed_frames": 2,
            "polygon": [(0, 0), (10, 0), (10, 10), (0, 10)],
            "rule": SignRule(
                sign_id=1,
                sign_label="3.28",
                restriction="no_parking",
                time_limit_s=300.0,
                applies_now=True,
            ),
        },
    }
    manager._states[1]["zone_mask"][20:70, 20:80] = 255
    manager.last_road_mask = np.ones((100, 120), dtype=np.uint8) * 255

    active_ids = manager._active_zone_ids(observed_ids)
    zones = manager._build_active_zones(active_ids)

    assert active_ids == {1}
    assert len(zones) == 1
    assert zones[0].metadata["sign_locked"] is False
    assert zones[0].metadata["sign_missed_frames"] == 0
    assert zones[0].metadata["last_sign_confidence"] == 0.72
    assert zones[0].metadata["current_sign_confidence"] == 0.28


def test_locked_sign_zone_is_not_active_when_detector_sees_no_sign() -> None:
    manager = SignZoneManager.__new__(SignZoneManager)
    manager._max_missing_frames = 30
    manager._sign_presence_confidence = 0.25
    manager._states = {
        1: {
            "locked": True,
            "missed_frames": 1,
            "polygon": [(20, 20), (80, 20), (80, 70), (20, 70)],
            "rule": SignRule(
                sign_id=0,
                sign_label="3.27",
                restriction="no_stopping",
                time_limit_s=0.0,
                applies_now=True,
            ),
            "current_confidence": 0.0,
        },
    }

    assert manager._active_zone_ids(set()) == set()


def test_locked_sign_keeps_plate_rule_when_plate_detector_drops_out() -> None:
    manager = SignZoneManager.__new__(SignZoneManager)
    previous = SignRule(
        sign_id=0,
        sign_label="3.27",
        restriction="no_stopping",
        time_limit_s=0.0,
        applies_now=True,
        start_mode="inside_zone",
        direction="both",
        plate_labels=["8.2.4"],
    )
    current_without_plate = SignRule(
        sign_id=0,
        sign_label="3.27",
        restriction="no_stopping",
        time_limit_s=0.0,
        applies_now=True,
    )

    selected = manager._select_effective_rule(
        {"locked": True, "rule": previous},
        current_without_plate,
    )

    assert selected is previous
