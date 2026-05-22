import math
import time

import cv2
import numpy as np

from perception_types import BoundingBox, Detection
from parking_vision.violation.road_zone_builder import (
    build_road_zone_by_station,
    build_projection_line_from_boundary,
    build_station_edge_model,
    build_station_map,
    build_station_projection_line,
    build_road_zone,
    build_projection_split,
    build_sign_side_mask,
    collect_road_profile,
    final_clip_mask,
    select_zone_mask_from_split,
    split_side_mask,
)
from parking_vision.violation.sign_rules import SignRule, group_sign_stacks
from parking_vision.violation.zone_manager import SignZoneManager


def _det(label_id: int, bbox: tuple[float, float, float, float], label: str | None = None) -> Detection:
    return Detection(
        module="sign",
        class_id=label_id,
        class_name=label or str(label_id),
        confidence=0.9,
        bbox=BoundingBox(*bbox),
        metadata={"sign_label": label} if label else {},
    )


def _rule(direction: str = "forward") -> SignRule:
    return SignRule(
        sign_id=0,
        sign_label="3.27",
        restriction="no_stopping",
        time_limit_s=0.0,
        applies_now=True,
        start_mode="inside_zone" if direction == "both" else "to_sign" if direction == "backward" else "from_sign",
        direction=direction,
    )


def _station_road() -> np.ndarray:
    road = np.zeros((220, 280), dtype=np.uint8)
    cv2.fillPoly(
        road,
        [np.asarray([(70, 20), (170, 20), (245, 200), (35, 200)], dtype=np.int32)],
        255,
    )
    return road


def _station_detection(x_shift: float = 0.0) -> Detection:
    return _det(0, (194 + x_shift, 72, 216 + x_shift, 98), "3.27")


def test_station_builder_does_not_use_physical_sign_base() -> None:
    road = _station_road()
    detection = _station_detection()

    result = build_road_zone_by_station(
        detection=detection,
        road_mask=road,
        rule=_rule("forward"),
        terminators=[],
        locked_side="right",
        locked_station_direction=-1,
        geometry_scale=1.0,
    )

    assert result is not None
    assert result.metadata["physical_base_estimation_used"] is False
    assert result.metadata["anchor_mode"] == "sign_x_to_edge_station"
    assert int(round(float(result.projected_sign_ground_point[1]))) != int(round(float(detection.bbox.y2)))


def test_station_builder_right_side_forward_split_and_opposite_side_outside() -> None:
    road = _station_road()

    result = build_road_zone_by_station(
        detection=_station_detection(),
        road_mask=road,
        rule=_rule("forward"),
        terminators=[],
        locked_side="right",
        locked_station_direction=-1,
        geometry_scale=1.0,
    )

    assert result is not None
    assert result.zone_mask[70, 185] > 0
    assert result.zone_mask[160, 225] == 0
    assert result.zone_mask[70, 115] == 0
    assert np.count_nonzero((result.zone_mask > 0) & (result.hard_road_mask == 0)) == 0


def test_station_builder_right_side_backward_for_8_2_3() -> None:
    road = _station_road()

    result = build_road_zone_by_station(
        detection=_station_detection(),
        road_mask=road,
        rule=_rule("backward"),
        terminators=[],
        locked_side="right",
        locked_station_direction=-1,
        geometry_scale=1.0,
    )

    assert result is not None
    assert result.zone_mask[160, 225] > 0
    assert result.zone_mask[70, 185] == 0


def test_station_builder_inside_plate_keeps_both_directions_same_side_only() -> None:
    road = _station_road()

    result = build_road_zone_by_station(
        detection=_station_detection(),
        road_mask=road,
        rule=_rule("both"),
        terminators=[],
        locked_side="right",
        locked_station_direction=-1,
        geometry_scale=1.0,
    )

    assert result is not None
    assert result.zone_mask[70, 185] > 0
    assert result.zone_mask[160, 225] > 0
    assert result.zone_mask[120, 100] == 0


def test_station_builder_metadata_reports_selected_split_part() -> None:
    road = _station_road()

    forward = build_road_zone_by_station(
        detection=_station_detection(),
        road_mask=road,
        rule=_rule("forward"),
        terminators=[],
        locked_side="right",
        locked_station_direction=-1,
        geometry_scale=1.0,
    )
    backward = build_road_zone_by_station(
        detection=_station_detection(),
        road_mask=road,
        rule=_rule("backward"),
        terminators=[],
        locked_side="right",
        locked_station_direction=-1,
        geometry_scale=1.0,
    )
    both = build_road_zone_by_station(
        detection=_station_detection(),
        road_mask=road,
        rule=_rule("both"),
        terminators=[],
        locked_side="right",
        locked_station_direction=-1,
        geometry_scale=1.0,
    )

    assert forward is not None and backward is not None and both is not None
    assert forward.metadata["selected_zone_part"] == "after"
    assert backward.metadata["selected_zone_part"] == "before"
    assert both.metadata["selected_zone_part"] == "both"
    assert both.metadata["inside_zone_plate"] is True
    assert forward.metadata["projection_line_source"] in {"station_cross_section", "inward_normal_fallback"}


def test_station_builder_sidewalk_false_positive_never_leaks() -> None:
    road = _station_road()
    sidewalk = np.zeros_like(road)
    sidewalk[:, 220:] = 255

    result = build_road_zone_by_station(
        detection=_station_detection(),
        road_mask=road,
        sidewalk_mask=sidewalk,
        rule=_rule("both"),
        terminators=[],
        locked_side="right",
        locked_station_direction=-1,
        geometry_scale=1.0,
    )

    assert result is not None
    assert np.count_nonzero((result.zone_mask > 0) & (result.hard_road_mask == 0)) == 0
    assert np.count_nonzero(result.zone_mask[:, 220:]) == 0


def test_station_builder_sign_jitter_keeps_station_stable() -> None:
    road = _station_road()
    locked_station = None
    locked_x = None
    stations = []
    projection_ys = []

    for jitter in (-5.0, 0.0, 5.0):
        result = build_road_zone_by_station(
            detection=_station_detection(jitter),
            road_mask=road,
            rule=_rule("forward"),
            terminators=[],
            locked_side="right",
            locked_sign_station=locked_station,
            locked_sign_x=locked_x,
            locked_station_direction=-1,
            geometry_scale=1.0,
        )
        assert result is not None
        locked_station = float(result.metadata["stable_sign_station"])
        locked_x = float(result.metadata["stable_sign_x"])
        stations.append(locked_station)
        line = result.metadata["projection_line"]
        projection_ys.append(0.5 * (float(line[0][1]) + float(line[1][1])))

    assert max(stations) - min(stations) < 10.0
    assert max(projection_ys) - min(projection_ys) < 12.0


def test_station_builder_downsample_refines_to_full_resolution_reasonably() -> None:
    road = _station_road()
    full = build_road_zone_by_station(
        detection=_station_detection(),
        road_mask=road,
        rule=_rule("forward"),
        terminators=[],
        locked_side="right",
        locked_station_direction=-1,
        geometry_scale=1.0,
    )
    small = build_road_zone_by_station(
        detection=_station_detection(),
        road_mask=road,
        rule=_rule("forward"),
        terminators=[],
        locked_side="right",
        locked_station_direction=-1,
        geometry_scale=0.5,
    )

    assert full is not None and small is not None
    assert np.linalg.norm(full.projected_sign_ground_point - small.projected_sign_ground_point) < 12.0
    assert np.count_nonzero((small.zone_mask > 0) & (small.hard_road_mask == 0)) == 0


def test_station_projection_cross_section_stays_inside_road_without_station_isoline() -> None:
    road = _station_road()
    side = build_road_zone_by_station(
        detection=_station_detection(),
        road_mask=road,
        rule=_rule("both"),
        terminators=[],
        locked_side="right",
        locked_station_direction=-1,
        geometry_scale=1.0,
    )
    assert side is not None
    edge = build_station_edge_model(hard_road=side.side_mask, side="right")
    assert edge is not None
    station_map = build_station_map(edge_model=edge, shape=road.shape)
    far_station_map = station_map + 10000.0

    line, fallback = build_station_projection_line(
        sign_side_mask=side.side_mask,
        station_map=far_station_map,
        stable_sign_station=float(side.metadata["stable_sign_station"]),
        stable_sign_station_anchor=side.projected_sign_ground_point,
        hard_road_mask=side.hard_road_mask,
        edge_model=edge,
    )

    assert fallback is False
    assert line is not None
    for x, y in line:
        assert side.hard_road_mask[int(round(y)), int(round(x))] > 0


def test_projection_split_uses_same_progress_for_masks_and_zone_selection() -> None:
    side = np.zeros((80, 140), dtype=np.uint8)
    side[20:70, 40:120] = 255
    hard = side.copy()
    station_map = np.repeat(np.arange(80, dtype=np.float32)[:, None], 140, axis=1)
    progress = station_map - 45.0
    edge = build_station_edge_model(hard_road=side, side="right")
    assert edge is not None

    split = build_projection_split(
        hard_road_mask=hard,
        sign_side_mask=side,
        signed_progress=progress,
        station_map=station_map,
        stable_sign_station=45.0,
        projection_anchor=np.asarray([119.0, 45.0], dtype=np.float32),
        traffic_dir=np.asarray([0.0, 1.0], dtype=np.float32),
        side="right",
        edge_model=edge,
    )

    assert split is not None
    assert split.before_sign_mask[30, 80] > 0
    assert split.after_sign_mask[60, 80] > 0
    forward, part = select_zone_mask_from_split(
        projection_split=split,
        signed_progress=progress,
        rule_direction="forward",
        length_px=None,
    )
    backward, back_part = select_zone_mask_from_split(
        projection_split=split,
        signed_progress=progress,
        rule_direction="backward",
        length_px=None,
    )
    assert forward is not None and backward is not None
    assert part == "after"
    assert back_part == "before"
    assert np.count_nonzero(forward[:43, :]) == 0
    assert np.count_nonzero(backward[47:, :]) == 0


def test_final_clip_mask_cannot_leak_outside_hard_road_after_cleaning() -> None:
    hard = np.zeros((80, 120), dtype=np.uint8)
    hard[20:60, 20:80] = 255
    mask = np.zeros_like(hard)
    mask[18:62, 18:82] = 255

    clipped = final_clip_mask(mask, hard, min_area=20)

    assert clipped is not None
    assert np.count_nonzero((clipped > 0) & (hard == 0)) == 0


def test_sign_side_mask_uses_full_interpolated_right_half_of_trapezoid() -> None:
    road = np.zeros((120, 160), dtype=np.uint8)
    cv2.fillPoly(road, [np.asarray([(30, 20), (110, 20), (145, 110), (15, 110)], dtype=np.int32)], 255)
    profile = collect_road_profile(road)

    side = build_sign_side_mask(
        road_mask=road,
        profile=profile,
        side="right",
        anchor_point=np.asarray([145.0, 110.0], dtype=np.float32),
        frame_w=160,
        frame_h=120,
    )

    assert side is not None
    ys, xs = np.where(road > 0)
    centers = {y: 0.5 * (xs[ys == y].min() + xs[ys == y].max()) for y in np.unique(ys)}
    left_hits = 0
    right_total = 0
    right_hits = 0
    for y, x in zip(ys, xs):
        if x < centers[y] - 1:
            left_hits += int(side[y, x] > 0)
        elif x > centers[y] + 1:
            right_total += 1
            right_hits += int(side[y, x] > 0)

    assert left_hits == 0
    assert right_hits / max(1, right_total) > 0.92


def test_collect_profile_ignores_disconnected_sidewalk_island() -> None:
    road = np.zeros((80, 180), dtype=np.uint8)
    road[20:70, 20:100] = 255
    road[30:45, 145:170] = 255

    profile = collect_road_profile(road)

    assert profile
    assert max(right for _, _, right in profile) <= 100


def test_projection_line_is_boundary_intersection_and_stable_under_anchor_jitter() -> None:
    side = np.zeros((120, 160), dtype=np.uint8)
    side[20:110, 70:145] = 255
    traffic_dir = np.asarray([0.0, -1.0], dtype=np.float32)

    lines = [
        build_projection_line_from_boundary(
            mask=side,
            anchor_point=np.asarray([145.0 + jitter, 70.0], dtype=np.float32),
            traffic_dir=traffic_dir,
            min_length=30.0,
        )
        for jitter in (-5.0, 0.0, 5.0)
    ]

    assert all(line is not None for line in lines)
    angles = []
    for line in lines:
        p0 = np.asarray(line[0], dtype=np.float32)
        p1 = np.asarray(line[1], dtype=np.float32)
        vec = p1 - p0
        angles.append(math.degrees(math.atan2(float(vec[1]), float(vec[0]))))
    assert max(angles) - min(angles) <= 2.0


def test_split_modes_keep_expected_progress_side_only() -> None:
    side = np.zeros((80, 140), dtype=np.uint8)
    side[20:70, 40:120] = 255
    anchor = np.asarray([80.0, 45.0], dtype=np.float32)
    traffic = np.asarray([1.0, 0.0], dtype=np.float32)

    forward = split_side_mask(side_mask=side, anchor_point=anchor, traffic_dir=traffic, rule_direction="forward")
    backward = split_side_mask(side_mask=side, anchor_point=anchor, traffic_dir=traffic, rule_direction="backward")
    both = split_side_mask(side_mask=side, anchor_point=anchor, traffic_dir=traffic, rule_direction="both")

    assert forward is not None and backward is not None and both is not None
    assert np.count_nonzero(forward[:, :76]) == 0
    assert np.count_nonzero(backward[:, 84:]) == 0
    assert np.count_nonzero(both[:, :76]) > 0
    assert np.count_nonzero(both[:, 84:]) > 0


def test_build_road_zone_hard_clips_sidewalk_and_runs_quickly() -> None:
    road = np.zeros((180, 260), dtype=np.uint8)
    road[40:170, 40:210] = 255
    sidewalk = np.zeros_like(road)
    sidewalk[:, 175:230] = 255
    detection = _det(0, (208, 35, 228, 65), "3.27")

    start = time.perf_counter()
    result = build_road_zone(
        detection=detection,
        road_mask=road,
        sidewalk_mask=sidewalk,
        rule=_rule("forward"),
        terminators=[],
        locked_side="right",
        locked_dir=np.asarray([0.0, -1.0], dtype=np.float32),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert result is not None
    assert np.count_nonzero((result.zone_mask > 0) & (result.hard_road_mask == 0)) == 0
    assert np.count_nonzero(result.zone_mask[:, 175:]) == 0
    assert elapsed_ms < 20.0


def test_road_geometry_does_not_depend_on_car_detections() -> None:
    manager = SignZoneManager.__new__(SignZoneManager)
    road = np.zeros((80, 120), dtype=np.uint8)
    road[20:70, 20:100] = 255
    car = Detection("car", 0, "car", 0.9, BoundingBox(30, 30, 80, 65), track_id=1)

    no_car = manager._prepare_road_geometry_mask(road, [])
    with_car = manager._prepare_road_geometry_mask(road, [car])

    assert np.array_equal(no_car, with_car)


def test_two_stacked_same_class_signs_keep_separate_plate_stacks() -> None:
    upper = _det(0, (100, 20, 130, 50), "3.27")
    upper_plate = _det(10, (101, 54, 129, 82), "8.2.4")
    lower = _det(0, (100, 95, 130, 125), "3.27")
    lower_plate = _det(9, (101, 129, 129, 157), "8.2.3")

    stacks = group_sign_stacks([upper, lower, upper_plate, lower_plate])

    assert len(stacks) == 2
    assert stacks[0].main is upper
    assert stacks[0].plate_labels == ["8.2.4"]
    assert stacks[1].main is lower
    assert stacks[1].plate_labels == ["8.2.3"]
