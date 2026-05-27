from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from perception_types import Detection

from .sign_rules import SignRule, is_zone_terminator


GEOMETRY_VERSION = "road_mask_split_v2"


@dataclass(slots=True)
class RoadZoneBuildResult:
    polygon: list[tuple[int, int]]
    zone_mask: np.ndarray
    side_mask: np.ndarray
    side: str
    traffic_dir: np.ndarray
    tangent: np.ndarray
    inward_normal: np.ndarray
    projected_sign_ground_point: np.ndarray
    raw_projected_sign_ground_point: np.ndarray
    projection_line: list[list[float]]
    confidence: float
    metadata: dict[str, Any]


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    road = np.where(mask > 0, 255, 0).astype(np.uint8)
    if road.size == 0:
        return road
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    road = cv2.morphologyEx(road, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return road


def build_road_zone(
    *,
    detection: Detection,
    road_mask: np.ndarray,
    rule: SignRule,
    terminators: list[Detection],
    locked_side: str | None = None,
    locked_anchor: np.ndarray | None = None,
    locked_dir: np.ndarray | None = None,
    max_anchor_jump_px: float = 90.0,
) -> RoadZoneBuildResult | None:
    road = normalize_mask(road_mask)
    if road.size == 0 or int(np.count_nonzero(road)) < 80:
        return None

    frame_h, frame_w = road.shape[:2]
    profile = collect_road_profile(road)
    if len(profile) < 8:
        return None

    raw_anchor_info = estimate_anchor_on_road_edge(
        detection=detection,
        profile=profile,
        frame_w=frame_w,
        frame_h=frame_h,
        locked_side=locked_side,
    )
    if raw_anchor_info is None:
        return None

    raw_y, raw_side, raw_x, anchor_confidence = raw_anchor_info
    side = raw_side
    if locked_side in ("left", "right"):
        side = locked_side
        if raw_side != locked_side:
            locked_info = estimate_anchor_on_road_edge(
                detection=detection,
                profile=profile,
                frame_w=frame_w,
                frame_h=frame_h,
                locked_side=locked_side,
            )
            if locked_info is not None:
                raw_y, raw_side, raw_x, anchor_confidence = locked_info

    raw_anchor = np.asarray([float(raw_x), float(raw_y)], dtype=np.float32)
    anchor = stabilize_anchor(
        raw_anchor=raw_anchor,
        locked_anchor=locked_anchor,
        side=side,
        profile=profile,
        max_jump_px=max_anchor_jump_px,
    )
    if anchor is None:
        return None
    anchor_jump_px = None
    if locked_anchor is not None:
        anchor_jump_px = float(np.linalg.norm(raw_anchor - locked_anchor.astype(np.float32)))
        if anchor_jump_px > max_anchor_jump_px:
            anchor_confidence *= 0.45

    tangent, inward_normal, road_width = estimate_local_road_vectors(
        profile=profile,
        anchor_y=int(round(float(anchor[1]))),
        side=side,
        frame_w=frame_w,
        frame_h=frame_h,
    )
    if tangent is None or inward_normal is None:
        return None

    traffic_dir, direction_confidence = choose_right_hand_traffic_dir(
        tangent=tangent,
        inward_normal=inward_normal,
        side=side,
        locked_dir=locked_dir,
    )
    side_mask = build_sign_side_mask(
        road_mask=road,
        profile=profile,
        side=side,
        anchor_point=anchor,
        frame_w=frame_w,
        frame_h=frame_h,
    )
    if side_mask is None or int(np.count_nonzero(side_mask)) < 80:
        return None

    distance_px, distance_estimated = estimate_distance_px(rule.distance_m, frame_w, frame_h)
    progress_limit_px = distance_px
    progress_limit_px = limit_by_terminators(
        length_px=progress_limit_px,
        anchor_point=anchor,
        traffic_dir=traffic_dir,
        side=side,
        road_mask=road,
        profile=profile,
        terminators=terminators,
        mode=rule.direction,
    )

    zone_mask = split_side_mask(
        side_mask=side_mask,
        anchor_point=anchor,
        traffic_dir=traffic_dir,
        rule_direction=rule.direction,
        length_px=progress_limit_px,
    )
    if zone_mask is None or int(np.count_nonzero(zone_mask)) < 40:
        return None
    zone_mask = clean_mask(cv2.bitwise_and(zone_mask, road))

    polygon = mask_to_polygon(zone_mask)
    if polygon is None:
        return None

    projection_line = build_projection_line(
        mask=side_mask,
        anchor_point=anchor,
        line_dir=inward_normal,
        min_length=max(32.0, road_width * 0.18),
    )
    if projection_line is None:
        projection_line = [
            clamp_point(anchor, frame_w, frame_h),
            clamp_point(anchor + inward_normal * float(np.clip(road_width * 0.55, 42.0, max(frame_w, frame_h))), frame_w, frame_h),
        ]

    road_support = mask_support(zone_mask, road)
    side_area = max(1, int(np.count_nonzero(side_mask)))
    selected_ratio = int(np.count_nonzero(zone_mask)) / float(side_area)
    confidence = float(np.clip(
        0.42 + 0.22 * road_support + 0.14 * min(1.0, selected_ratio * 2.0)
        + 0.12 * anchor_confidence + 0.10 * direction_confidence,
        0.0,
        0.98,
    ))

    metadata: dict[str, Any] = {
        "zone_source": "road_mask_split_components",
        "zone_confidence": round(confidence, 3),
        "road_mask_support": round(road_support, 3),
        "geometry_version": GEOMETRY_VERSION,
        "rule_direction": rule.direction,
        "rule_start_mode": rule.start_mode,
        "side": side,
        "anchor_confidence": round(float(anchor_confidence), 3),
        "direction_confidence": "low" if direction_confidence < 0.25 else "high",
        "direction_confidence_score": round(float(direction_confidence), 3),
        "projected_sign_ground_point": [float(anchor[0]), float(anchor[1])],
        "raw_projected_sign_ground_point": [float(raw_anchor[0]), float(raw_anchor[1])],
        "projection_line": projection_line,
        "zone_direction_vec": [float(traffic_dir[0]), float(traffic_dir[1])],
        "right_hand_traffic": True,
        "side_mask_area": int(np.count_nonzero(side_mask)),
        "split_component_area": int(np.count_nonzero(zone_mask)),
        "distance_px_estimated": bool(distance_estimated),
    }
    if progress_limit_px is not None:
        metadata["distance_limit_px"] = round(float(progress_limit_px), 2)
    if anchor_jump_px is not None:
        metadata["raw_anchor_jump_px"] = round(anchor_jump_px, 2)

    return RoadZoneBuildResult(
        polygon=polygon,
        zone_mask=zone_mask,
        side_mask=side_mask,
        side=side,
        traffic_dir=traffic_dir,
        tangent=tangent,
        inward_normal=inward_normal,
        projected_sign_ground_point=anchor,
        raw_projected_sign_ground_point=raw_anchor,
        projection_line=projection_line,
        confidence=confidence,
        metadata=metadata,
    )


def collect_road_profile(road_mask: np.ndarray) -> list[tuple[int, int, int]]:
    height = road_mask.shape[0]
    step = max(2, int(round(height / 180)))
    profile: list[tuple[int, int, int]] = []
    for y in range(height - 1, -1, -step):
        edges = get_road_edges(road_mask, y, band=max(2, step))
        if edges is None:
            continue
        left, right = edges
        if right - left < 24:
            continue
        profile.append((y, left, right))
    profile.reverse()
    return profile


def get_road_edges(road_mask: np.ndarray, y: int, band: int = 4) -> tuple[int, int] | None:
    height, _ = road_mask.shape[:2]
    if y < 0 or y >= height:
        return None
    y1 = max(0, y - band)
    y2 = min(height, y + band + 1)
    row_slice = road_mask[y1:y2, :]
    road_indices = np.where(np.any(row_slice > 0, axis=0))[0]
    if len(road_indices) < 2:
        return None
    return int(road_indices[0]), int(road_indices[-1])


def estimate_anchor_on_road_edge(
    *,
    detection: Detection,
    profile: list[tuple[int, int, int]],
    frame_w: int,
    frame_h: int,
    locked_side: str | None = None,
) -> tuple[int, str, int, float] | None:
    x1, y1, x2, y2 = detection.bbox.to_int_tuple()
    sign_x = 0.5 * float(x1 + x2)
    sign_h = max(1.0, float(y2 - y1))
    search_top = max(0, y2 - int(0.20 * sign_h))
    search_bottom = min(frame_h - 1, y2 + int(max(frame_h * 0.56, sign_h * 8.0)))
    preferred_y = min(search_bottom, y2 + int(max(frame_h * 0.18, sign_h * 3.0)))

    rows = [(y, left, right) for y, left, right in profile if search_top <= y <= search_bottom]
    if not rows:
        rows = [(y, left, right) for y, left, right in profile if abs(y - y2) <= int(frame_h * 0.34)]
    if not rows:
        return None

    best_score = float("inf")
    best: tuple[int, str, int, float] | None = None
    for y, left, right in rows:
        road_width = max(1.0, float(right - left))
        if locked_side in ("left", "right"):
            preferred_side = locked_side
            sides = [locked_side]
        elif sign_x <= left:
            preferred_side = "left"
            sides = ["left", "right"]
        elif sign_x >= right:
            preferred_side = "right"
            sides = ["right", "left"]
        else:
            preferred_side = "left" if abs(sign_x - left) <= abs(sign_x - right) else "right"
            sides = ["left", "right"]

        for side in sides:
            edge_x = int(right if side == "right" else left)
            edge_dist = abs(sign_x - float(edge_x))
            if edge_dist > max(180.0, min(frame_w * 0.65, road_width * 0.95)):
                continue
            side_penalty = 0.0 if side == preferred_side else min(95.0, 0.45 * road_width)
            y_penalty = 0.12 * abs(float(y - preferred_y))
            projection_penalty = 0.025 * float(np.hypot(sign_x - edge_x, y - y2))
            outside_bonus = 0.0
            if side == "right" and sign_x >= right:
                outside_bonus = min(38.0, 0.16 * edge_dist)
            elif side == "left" and sign_x <= left:
                outside_bonus = min(38.0, 0.16 * edge_dist)
            score = edge_dist + y_penalty + side_penalty + projection_penalty - outside_bonus
            confidence = float(np.clip(1.0 - score / max(180.0, frame_w * 0.35), 0.05, 1.0))
            if score < best_score:
                best_score = score
                best = (int(y), side, int(edge_x), confidence)

    return best


def stabilize_anchor(
    *,
    raw_anchor: np.ndarray,
    locked_anchor: np.ndarray | None,
    side: str,
    profile: list[tuple[int, int, int]],
    max_jump_px: float,
) -> np.ndarray | None:
    if locked_anchor is None:
        return raw_anchor.astype(np.float32)

    locked = locked_anchor.astype(np.float32)
    jump = float(np.linalg.norm(raw_anchor - locked))
    if jump > max_jump_px:
        return project_point_to_side_boundary(locked, side, profile)

    alpha = 0.22 if jump > 12.0 else 0.08
    target = (1.0 - alpha) * locked + alpha * raw_anchor.astype(np.float32)
    return project_point_to_side_boundary(target, side, profile)


def project_point_to_side_boundary(
    point: np.ndarray,
    side: str,
    profile: list[tuple[int, int, int]],
) -> np.ndarray | None:
    best_dist = float("inf")
    best: np.ndarray | None = None
    for y, left, right in profile:
        edge_x = right if side == "right" else left
        dist = abs(float(y) - float(point[1])) + 0.18 * abs(float(edge_x) - float(point[0]))
        if dist < best_dist:
            best_dist = dist
            best = np.asarray([float(edge_x), float(y)], dtype=np.float32)
    return best


def estimate_local_road_vectors(
    *,
    profile: list[tuple[int, int, int]],
    anchor_y: int,
    side: str,
    frame_w: int,
    frame_h: int,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    rows = [(y, left, right) for y, left, right in profile if abs(y - anchor_y) <= max(36, int(frame_h * 0.14))]
    if len(rows) < 3:
        rows = profile
    if len(rows) < 3:
        return None, None, float(frame_w * 0.30)

    centers = np.asarray([(0.5 * (left + right), float(y)) for y, left, right in rows], dtype=np.float32)
    edge_points = np.asarray([(float(right if side == "right" else left), float(y)) for y, left, right in rows], dtype=np.float32)
    widths = [float(right - left) for _, left, right in rows]

    centered = centers - np.mean(centers, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    tangent = normalize_vec(vecs[:, int(np.argmax(vals))].astype(np.float32))

    normal_a = normalize_vec(np.asarray([-tangent[1], tangent[0]], dtype=np.float32))
    normal_b = -normal_a
    edge_center = np.mean(edge_points, axis=0)
    road_center = np.mean(centers, axis=0)
    inward_normal = normal_a if np.linalg.norm(edge_center + normal_a * 30.0 - road_center) < np.linalg.norm(edge_center + normal_b * 30.0 - road_center) else normal_b
    return tangent, normalize_vec(inward_normal), float(np.median(widths) if widths else frame_w * 0.30)


def choose_right_hand_traffic_dir(
    *,
    tangent: np.ndarray,
    inward_normal: np.ndarray,
    side: str,
    locked_dir: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    base = normalize_vec(tangent.astype(np.float32))
    candidates = [base, -base]

    def rotate_left(v: np.ndarray) -> np.ndarray:
        return np.asarray([v[1], -v[0]], dtype=np.float32)

    def rotate_right(v: np.ndarray) -> np.ndarray:
        return np.asarray([-v[1], v[0]], dtype=np.float32)

    scores = []
    inward = normalize_vec(inward_normal.astype(np.float32))
    for candidate in candidates:
        driver_left_or_right = rotate_left(candidate) if side == "right" else rotate_right(candidate)
        scores.append(float(np.dot(normalize_vec(driver_left_or_right), inward)))

    order = np.argsort(scores)
    best_idx = int(order[-1])
    margin = float(scores[best_idx] - scores[int(order[-2])])
    direction_confidence = float(np.clip((margin + 0.05) / 1.05, 0.0, 1.0))

    if direction_confidence < 0.18 and locked_dir is not None:
        locked = normalize_vec(locked_dir.astype(np.float32))
        if float(np.linalg.norm(locked)) > 0.5:
            return locked, direction_confidence

    chosen = candidates[best_idx]
    if locked_dir is not None and direction_confidence < 0.35:
        locked = normalize_vec(locked_dir.astype(np.float32))
        if float(np.dot(chosen, locked)) < 0.0:
            chosen = -chosen
    return normalize_vec(chosen), direction_confidence


def build_sign_side_mask(
    *,
    road_mask: np.ndarray,
    profile: list[tuple[int, int, int]],
    side: str,
    anchor_point: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> np.ndarray | None:
    side_mask = np.zeros_like(road_mask, dtype=np.uint8)
    for y, left, right in profile:
        if y < 0 or y >= frame_h:
            continue
        mid = int(round(0.5 * (left + right)))
        if side == "right":
            x1, x2 = max(0, mid), min(frame_w - 1, right)
        else:
            x1, x2 = max(0, left), min(frame_w - 1, mid)
        if x2 < x1:
            continue
        side_mask[y, x1:x2 + 1] = np.where(road_mask[y, x1:x2 + 1] > 0, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    side_mask = cv2.dilate(side_mask, kernel, iterations=2)
    side_mask = cv2.bitwise_and(side_mask, road_mask)

    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats((side_mask > 0).astype(np.uint8), 8)
    if labels_count <= 1:
        return None

    best_label = 0
    best_score = -float("inf")
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 40:
            continue
        component = labels == label
        ys, xs = np.where(component)
        if xs.size == 0:
            continue
        min_dist = float(np.min(np.hypot(xs.astype(np.float32) - anchor_point[0], ys.astype(np.float32) - anchor_point[1])))
        centroid = np.asarray(centroids[label], dtype=np.float32)
        centroid_dist = float(np.linalg.norm(centroid - anchor_point))
        score = -0.90 * min_dist - 0.015 * centroid_dist + 0.004 * area
        if min_dist <= 18.0:
            score += 80.0
        if score > best_score:
            best_score = score
            best_label = label

    if best_label == 0:
        return None
    selected = np.zeros_like(side_mask, dtype=np.uint8)
    selected[labels == best_label] = 255
    return clean_mask(selected)


def split_side_mask(
    *,
    side_mask: np.ndarray,
    anchor_point: np.ndarray,
    traffic_dir: np.ndarray,
    rule_direction: str,
    length_px: float | None = None,
) -> np.ndarray | None:
    ys, xs = np.where(side_mask > 0)
    if xs.size < 40:
        return None
    points = np.column_stack((xs, ys)).astype(np.float32)
    progress = (points - anchor_point.astype(np.float32)) @ normalize_vec(traffic_dir.astype(np.float32))

    if rule_direction == "both":
        keep = np.ones(progress.shape, dtype=bool)
        if length_px is not None:
            keep &= np.abs(progress) <= float(length_px) + 2.0
    elif rule_direction == "backward":
        keep = progress <= 2.0
        if length_px is not None:
            keep &= progress >= -float(length_px) - 2.0
    else:
        keep = progress >= -2.0
        if length_px is not None:
            keep &= progress <= float(length_px) + 2.0

    if int(np.count_nonzero(keep)) < 40:
        return None
    selected = np.zeros_like(side_mask, dtype=np.uint8)
    kept = points[keep].astype(np.int32)
    selected[kept[:, 1], kept[:, 0]] = 255
    return keep_components_near_anchor(selected, anchor_point, traffic_dir, rule_direction)


def keep_components_near_anchor(
    mask: np.ndarray,
    anchor_point: np.ndarray,
    traffic_dir: np.ndarray,
    rule_direction: str,
) -> np.ndarray | None:
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if labels_count <= 1:
        return None
    direction = normalize_vec(traffic_dir.astype(np.float32))
    selected = np.zeros_like(mask, dtype=np.uint8)
    for mode in (["forward", "backward"] if rule_direction == "both" else [rule_direction]):
        best_label = 0
        best_score = -float("inf")
        for label in range(1, labels_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 40:
                continue
            centroid = np.asarray(centroids[label], dtype=np.float32)
            progress = float((centroid - anchor_point) @ direction)
            if mode == "forward" and progress < -10.0:
                continue
            if mode == "backward" and progress > 10.0:
                continue
            score = 0.006 * area - 0.040 * abs(progress)
            if score > best_score:
                best_score = score
                best_label = label
        if best_label != 0:
            selected[labels == best_label] = 255

    if int(np.count_nonzero(selected)) < 40:
        return None
    return clean_mask(selected)


def estimate_distance_px(distance_m: float | None, frame_w: int, frame_h: int) -> tuple[float | None, bool]:
    if distance_m is None:
        return None, False
    diag = float(np.hypot(frame_w, frame_h))
    px = float(np.clip((float(distance_m) / 25.0) * (0.35 * diag), 80.0, 0.90 * diag))
    return px, True


def limit_by_terminators(
    *,
    length_px: float | None,
    anchor_point: np.ndarray,
    traffic_dir: np.ndarray,
    side: str,
    road_mask: np.ndarray,
    profile: list[tuple[int, int, int]],
    terminators: list[Detection],
    mode: str,
) -> float | None:
    candidates: list[float] = []
    direction = normalize_vec(traffic_dir.astype(np.float32))
    for terminator in terminators:
        if not is_zone_terminator(terminator):
            continue
        anchor_info = estimate_anchor_on_road_edge(
            detection=terminator,
            profile=profile,
            frame_w=road_mask.shape[1],
            frame_h=road_mask.shape[0],
            locked_side=side,
        )
        if anchor_info is None:
            continue
        y, term_side, x, _ = anchor_info
        if term_side != side:
            continue
        progress = float((np.asarray([float(x), float(y)], dtype=np.float32) - anchor_point) @ direction)
        candidate = -progress if mode == "backward" else progress
        if mode == "both":
            candidate = abs(progress)
        if candidate > 24.0:
            candidates.append(candidate)

    if not candidates:
        return length_px
    terminator_limit = min(candidates)
    if length_px is None:
        return terminator_limit
    return min(float(length_px), float(terminator_limit))


def build_projection_line(
    *,
    mask: np.ndarray,
    anchor_point: np.ndarray,
    line_dir: np.ndarray,
    min_length: float,
) -> list[list[float]] | None:
    if mask.size == 0:
        return None
    h, w = mask.shape[:2]
    direction = normalize_vec(line_dir.astype(np.float32))
    start = anchor_point.astype(np.float32)
    max_len = float(np.hypot(w, h))
    last_inside: np.ndarray | None = None
    gap = 0.0
    for dist in np.arange(0.0, max_len + 1.0, 2.0):
        candidate = start + direction * float(dist)
        x = int(round(float(candidate[0])))
        y = int(round(float(candidate[1])))
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        patch = mask[max(0, y - 2):min(h, y + 3), max(0, x - 2):min(w, x + 3)]
        inside = bool(patch.size and np.any(patch > 0))
        if inside:
            last_inside = candidate.astype(np.float32)
            gap = 0.0
        elif last_inside is not None:
            gap += 2.0
            if gap > 14.0:
                break
    if last_inside is None or float(np.linalg.norm(last_inside - start)) < min_length:
        return None
    return [clamp_point(start, w, h), clamp_point(last_inside, w, h)]


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


def mask_to_polygon(mask: np.ndarray) -> list[tuple[int, int]] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = [contour for contour in contours if cv2.contourArea(contour) >= 40.0]
    if not significant:
        return None
    points = np.vstack(significant).reshape(-1, 2).astype(np.int32)
    hull = cv2.convexHull(points).reshape(-1, 2)
    if len(hull) < 3:
        x, y, w, h = cv2.boundingRect(points.reshape(-1, 1, 2))
        hull = np.asarray([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], dtype=np.int32)
    perimeter = cv2.arcLength(hull.reshape(-1, 1, 2), True)
    approx = cv2.approxPolyDP(hull.reshape(-1, 1, 2), max(2.0, 0.012 * perimeter), True).reshape(-1, 2)
    if len(approx) < 3:
        approx = hull
    return [(int(x), int(y)) for x, y in approx]


def mask_support(zone_mask: np.ndarray | None, road_mask: np.ndarray) -> float:
    if zone_mask is None or zone_mask.size == 0 or road_mask.size == 0:
        return 0.0
    zone = zone_mask > 0
    area = int(np.count_nonzero(zone))
    if area == 0:
        return 0.0
    return float(np.clip(np.count_nonzero(zone & (road_mask > 0)) / float(area), 0.0, 1.0))


def clamp_point(point: np.ndarray, frame_w: int, frame_h: int) -> list[float]:
    return [
        float(np.clip(point[0], 0, max(0, frame_w - 1))),
        float(np.clip(point[1], 0, max(0, frame_h - 1))),
    ]


def normalize_vec(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.asarray([0.0, -1.0], dtype=np.float32)
    return (vector / norm).astype(np.float32)