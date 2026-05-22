import numpy as np

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
