from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import cv2
import numpy as np

from perception_types import Detection
from road import RoadSegmenter

from .constants import SIGN_TIME_LIMITS_S
from .geometry import polygon_bbox
from .sign_rules import (
    SignRule,
    build_rule,
    group_sign_stacks,
    normalize_sign_label,
)
from .types import SignZone


@dataclass(slots=True)
class ZoneCandidate:
    polygon: list[tuple[int, int]]
    side: str
    direction_vec: np.ndarray
    confidence: float
    source: str
    metadata: dict[str, Any]


class SignZoneManager:
    def __init__(
        self,
        parking_time_limit_s: float = 300.0,
        max_missing_frames: int = 20,
        warmup_frames: int = 8,
        use_road_mask_geometry: bool = False,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.segmenter = RoadSegmenter(update_every_n_frames=1, downscale=0.5)

        self._states: dict[int, dict[str, Any]] = {}
        self._next_id = 1
        self._warmup_frames = max(3, warmup_frames)
        self._max_missing_frames = max(1, max_missing_frames)
        self._use_road_mask_geometry = use_road_mask_geometry

        self.time_limits_s = dict(SIGN_TIME_LIMITS_S)
        self.time_limits_s[1] = float(parking_time_limit_s)
        self.time_limits_s[2] = float(parking_time_limit_s)
        self.time_limits_s[3] = float(parking_time_limit_s)

        self._now_provider = now_provider or datetime.now

        self._vehicle_observations: deque[tuple[float, float, float, float]] = deque(maxlen=220)
        self._track_points: dict[int, deque[tuple[float, float]]] = {}

        # Геометрия зоны
        self._lane_width_fraction = 0.34
        self._lane_min_width_px = 34
        self._lane_max_width_px = 260
        self._anchor_search_margin_px = 170
        self._anchor_search_height_ratio = 0.16

        # Стабилизация
        self._polygon_smoothing_alpha = 0.14
        self._polygon_outlier_alpha = 0.03
        self._min_polygon_iou_for_normal_update = 0.18

        # Длина зоны
        self._min_zone_length_px = 110
        self._max_zone_length_px_ratio = 0.78

    def _reset_scene(self) -> None:
        self._states.clear()
        self._vehicle_observations.clear()
        self._track_points.clear()
        self._next_id = 1

    # ---------- public ----------

    def build_zones(
        self,
        sign_detections: list[Detection],
        frame: np.ndarray,
        car_detections: list[Detection] | None = None,
    ) -> list[SignZone]:
        height, width = frame.shape[:2]
        road_mask = self.segmenter.get_road_mask(frame)

        if getattr(self.segmenter, "last_scene_cut", False):
            self._reset_scene()

        if car_detections:
            self._update_vehicle_observations(car_detections, width, height)

        stacks = group_sign_stacks(sign_detections)
        now = self._now_provider()

        active_ids: set[int] = set()

        for stack in stacks:
            detection = stack.main
            rule = build_rule(stack, self.time_limits_s.get(detection.class_id, 300.0), now)
            if not rule.creates_violation_zone:
                continue

            sign_id = self._find_sign_id(detection)
            active_ids.add(sign_id)

            if sign_id not in self._states:
                self._states[sign_id] = {
                    "state": "WARMUP",
                    "polygon_buffer": [],
                    "frozen_polygon": None,
                    "source_bbox": detection.bbox,
                    "class_id": detection.class_id,
                    "sign_label": normalize_sign_label(detection),
                    "missed_frames": 0,
                    "rule": rule,
                    "side": "unknown",
                    "direction_vec": None,
                    "zone_metadata": {},
                }

            state = self._states[sign_id]
            state["source_bbox"] = detection.bbox
            state["class_id"] = detection.class_id
            state["sign_label"] = rule.sign_label
            state["rule"] = rule
            state["missed_frames"] = 0

            locked_side = None if state["state"] == "WARMUP" else state.get("side")
            locked_dir = None if state["state"] == "WARMUP" else state.get("direction_vec")

            result = self._calc_raw_polygon(
                detection=detection,
                road_mask=road_mask,
                frame_w=width,
                frame_h=height,
                car_detections=car_detections or [],
                rule=rule,
                locked_side=locked_side,
                locked_dir=locked_dir,
            )
            if result is None:
                continue

            raw_polygon = result.polygon
            raw_side = result.side
            raw_dir = result.direction_vec
            state["zone_metadata"] = {
                "zone_source": result.source,
                "zone_confidence": round(float(result.confidence), 3),
                **result.metadata,
            }

            if state["state"] == "WARMUP":
                state["polygon_buffer"].append(raw_polygon)
                if state.get("side") == "unknown":
                    state["side"] = raw_side
                if state.get("direction_vec") is None:
                    state["direction_vec"] = raw_dir
                else:
                    state["direction_vec"] = self._normalize_vec(
                        0.5 * np.asarray(state["direction_vec"], dtype=np.float32)
                        + 0.5 * np.asarray(raw_dir, dtype=np.float32)
                    )
                state["frozen_polygon"] = self._average_polygons(state["polygon_buffer"])

                if len(state["polygon_buffer"]) >= self._warmup_frames:
                    state["state"] = "LOCKED"
                    state["side"] = raw_side
                    state["direction_vec"] = self._normalize_vec(np.asarray(state["direction_vec"], dtype=np.float32))

            else:
                prev = state.get("frozen_polygon")
                if prev is None:
                    state["frozen_polygon"] = raw_polygon
                else:
                    iou = self._polygon_bbox_iou(prev, raw_polygon)
                    alpha = self._polygon_smoothing_alpha if iou >= self._min_polygon_iou_for_normal_update else self._polygon_outlier_alpha
                    state["frozen_polygon"] = self._smooth_polygon(prev, raw_polygon, alpha)

                # side фиксируем после warmup
                state["side"] = state.get("side", raw_side)
                if raw_dir is not None and state.get("direction_vec") is not None:
                    state["direction_vec"] = self._normalize_vec(
                        0.85 * np.asarray(state["direction_vec"], dtype=np.float32)
                        + 0.15 * np.asarray(raw_dir, dtype=np.float32)
                    )

        zones: list[SignZone] = []
        keys_to_delete: list[int] = []

        for sign_id, state in self._states.items():
            if sign_id not in active_ids:
                state["missed_frames"] += 1
                if state["missed_frames"] > self._max_missing_frames:
                    keys_to_delete.append(sign_id)
                continue

            polygon = state.get("frozen_polygon")
            rule = state.get("rule")
            if polygon is None or rule is None:
                continue

            x1, y1, x2, y2 = polygon_bbox(polygon)
            zones.append(
                SignZone(
                    sign_id=state["class_id"],
                    sign_label=state["sign_label"],
                    polygon=polygon,
                    source_bbox=state["source_bbox"],
                    time_limit_s=rule.time_limit_s,
                    restriction=rule.restriction,
                    applies_now=rule.applies_now,
                    side=state.get("side", "unknown"),
                    direction=rule.direction,
                    plate_labels=list(rule.plate_labels),
                    metadata={**dict(rule.metadata), **dict(state.get("zone_metadata", {}))},
                    _bbox_x1=x1,
                    _bbox_y1=y1,
                    _bbox_x2=x2,
                    _bbox_y2=y2,
                )
            )

        for key in keys_to_delete:
            del self._states[key]

        return zones

    @staticmethod
    def find_zone_for_car(car_detection: Detection, zones: list[SignZone]) -> SignZone | None:
        bbox = car_detection.bbox
        test_points = [
            (0.50 * (bbox.x1 + bbox.x2), bbox.y2),
            (bbox.x1 + 0.30 * (bbox.x2 - bbox.x1), bbox.y2 - 2),
            (bbox.x1 + 0.50 * (bbox.x2 - bbox.x1), bbox.y2 - 4),
            (bbox.x1 + 0.70 * (bbox.x2 - bbox.x1), bbox.y2 - 2),
            (bbox.x1 + 0.50 * (bbox.x2 - bbox.x1), bbox.y1 + 0.82 * (bbox.y2 - bbox.y1)),
        ]
        for zone in zones:
            for x, y in test_points:
                if zone.contains_point(float(x), float(y)):
                    return zone
        return None

    # ---------- polygon building ----------

    def _calc_raw_polygon(
        self,
        detection: Detection,
        road_mask: np.ndarray,
        frame_w: int,
        frame_h: int,
        car_detections: list[Detection],
        rule: SignRule | None = None,
        locked_side: str | None = None,
        locked_dir: np.ndarray | None = None,
    ) -> ZoneCandidate | None:
        scene_polygon, scene_side, scene_dir = self._scene_geometry_polygon(
            detection=detection,
            rule=rule,
            frame_w=frame_w,
            frame_h=frame_h,
            locked_side=locked_side,
            locked_dir=locked_dir,
        )
        scene_confidence = self._score_scene_geometry(
            polygon=scene_polygon,
            side=scene_side,
            direction=scene_dir,
            road_mask=road_mask,
            car_detections=car_detections,
            frame_w=frame_w,
            frame_h=frame_h,
            locked_dir=locked_dir,
        )
        scene_candidate = ZoneCandidate(
            polygon=scene_polygon,
            side=scene_side,
            direction_vec=scene_dir,
            confidence=scene_confidence,
            source="sign_direction",
            metadata={
                "rule_direction": rule.direction if rule is not None else "forward",
                "motion_track_count": self._moving_track_count(frame_h),
                "road_mask_support": round(self._polygon_mask_support(scene_polygon, road_mask), 3),
            },
        )

        road_candidate = self._build_road_mask_candidate(
            detection=detection,
            road_mask=road_mask,
            frame_w=frame_w,
            frame_h=frame_h,
            car_detections=car_detections,
            rule=rule,
            locked_side=locked_side,
            locked_dir=locked_dir,
        )
        best = scene_candidate
        if (
            self._use_road_mask_geometry
            and
            road_candidate is not None
            and road_candidate.metadata.get("road_mask_support", 0.0) >= 0.45
            and road_candidate.confidence >= scene_candidate.confidence + 0.18
        ):
            best = road_candidate
        if best.confidence >= 0.22:
            return best

        fallback_polygon = self._fallback_camera_polygon(detection, frame_w, frame_h)
        return ZoneCandidate(
            polygon=fallback_polygon,
            side=self._naive_side(detection, frame_w),
            direction_vec=np.array([0.0, -1.0], dtype=np.float32),
            confidence=0.18,
            source="fallback",
            metadata={"fallback_reason": "low_hybrid_confidence"},
        )

    def _build_road_mask_candidate(
        self,
        detection: Detection,
        road_mask: np.ndarray,
        frame_w: int,
        frame_h: int,
        car_detections: list[Detection],
        rule: SignRule | None = None,
        locked_side: str | None = None,
        locked_dir: np.ndarray | None = None,
    ) -> ZoneCandidate | None:
        profile = self._collect_road_profile(road_mask)
        if len(profile) < 8:
            return None

        anchor = self._estimate_anchor_on_road_edge(
            detection=detection,
            profile=profile,
            frame_h=frame_h,
            locked_side=locked_side,
        )
        if anchor is None:
            return None

        anchor_y, side, anchor_outer_x = anchor
        tangent, inward_normal, road_width = self._estimate_local_edge_vectors(
            profile=profile,
            anchor_y=anchor_y,
            side=side,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        if tangent is None or inward_normal is None:
            return None

        lane_width = int(np.clip(
            road_width * self._lane_width_fraction,
            self._lane_min_width_px,
            min(self._lane_max_width_px, max(road_width - 4, self._lane_min_width_px)),
        ))

        _, _, _, sign_y2 = detection.bbox.to_int_tuple()

        # Старт зоны: у основания знака, чуть ниже нижней границы знака
        anchor_point = np.array([
            float(anchor_outer_x),
            float(np.clip(sign_y2 + 2.0, 0, frame_h - 1)),
        ], dtype=np.float32)

        # Кандидаты: вдоль борта дороги в обе стороны.
        candidate_dirs = [tangent.copy(), -tangent.copy()]
        if locked_dir is not None:
            locked_dir = self._normalize_vec(np.asarray(locked_dir, dtype=np.float32))
            candidate_dirs.sort(key=lambda d: float(np.dot(self._normalize_vec(d), locked_dir)), reverse=True)

        best_payload: tuple[float, list[tuple[int, int]], np.ndarray] | None = None

        for direction in candidate_dirs:
            length_px = self._estimate_zone_length(
                anchor_point=anchor_point,
                direction=direction,
                inward_normal=inward_normal,
                lane_width=lane_width,
                road_mask=road_mask,
                frame_w=frame_w,
                frame_h=frame_h,
                distance_m=rule.distance_m if rule is not None else None,
            )

            polygon = self._build_polygon(
                anchor_point=anchor_point,
                direction=direction,
                inward_normal=inward_normal,
                near_width=lane_width,
                far_width=max(int(lane_width * 0.86), self._lane_min_width_px),
                length_px=length_px,
                frame_w=frame_w,
                frame_h=frame_h,
            )

            score = self._score_candidate_polygon(
                polygon=polygon,
                car_detections=car_detections,
                road_mask=road_mask,
                direction=direction,
                anchor_point=anchor_point,
                inward_normal=inward_normal,
                lane_width=lane_width,
                frame_w=frame_w,
                frame_h=frame_h,
            )

            if rule is not None:
                if rule.direction == "forward":
                    score += 3.0 if self._normalize_vec(direction)[1] < -0.05 else -3.0
                elif rule.direction == "backward":
                    score += 3.0 if self._normalize_vec(direction)[1] > 0.05 else -3.0

            if locked_dir is not None:
                score += 1.8 * max(0.0, float(np.dot(self._normalize_vec(direction), locked_dir)))

            if best_payload is None or score > best_payload[0]:
                best_payload = (score, polygon, direction)

        if best_payload is None:
            return None

        best_score, best_polygon, best_direction = best_payload
        mask_support = self._polygon_mask_support(best_polygon, road_mask)
        confidence = float(np.clip(0.18 + 0.045 * best_score + 0.42 * mask_support, 0.0, 0.94))
        return ZoneCandidate(
            polygon=best_polygon,
            side=side,
            direction_vec=self._normalize_vec(best_direction),
            confidence=confidence,
            source="road_mask",
            metadata={
                "rule_direction": rule.direction if rule is not None else "forward",
                "road_profile_rows": len(profile),
                "road_score": round(float(best_score), 3),
                "road_mask_support": round(mask_support, 3),
            },
        )

    def _build_polygon(
        self,
        anchor_point: np.ndarray,
        direction: np.ndarray,
        inward_normal: np.ndarray,
        near_width: int,
        far_width: int,
        length_px: float,
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[int, int]]:
        direction = self._normalize_vec(direction)
        inward_normal = self._normalize_vec(inward_normal)

        near_outer = anchor_point.astype(np.float32)
        near_inner = near_outer + inward_normal * float(near_width)

        far_outer = near_outer + direction * float(length_px)
        far_inner = far_outer + inward_normal * float(far_width)

        points = [near_outer, far_outer, far_inner, near_inner]
        result: list[tuple[int, int]] = []
        for pt in points:
            x = int(round(float(np.clip(pt[0], 0, frame_w - 1))))
            y = int(round(float(np.clip(pt[1], 0, frame_h - 1))))
            result.append((x, y))
        return result

    def _estimate_zone_length(
        self,
        anchor_point: np.ndarray,
        direction: np.ndarray,
        inward_normal: np.ndarray,
        lane_width: int,
        road_mask: np.ndarray,
        frame_w: int,
        frame_h: int,
        distance_m: float | None = None,
    ) -> float:
        if distance_m is not None:
            # Без калибровки только грубый визуальный proxy.
            # 25м -> ~35% диагонали кадра
            diag = float(np.hypot(frame_w, frame_h))
            px = np.clip((distance_m / 25.0) * (0.35 * diag), self._min_zone_length_px, 0.90 * diag)
            return float(px)

        direction = self._normalize_vec(direction)
        inward_normal = self._normalize_vec(inward_normal)

        max_len = float(max(frame_w, frame_h) * self._max_zone_length_px_ratio)
        step = 10.0
        best_len = float(self._min_zone_length_px)

        for length in np.arange(step, max_len + step, step):
            center = anchor_point + direction * length + inward_normal * (0.52 * lane_width)
            x = int(round(center[0]))
            y = int(round(center[1]))

            if x < 1 or x >= frame_w - 1 or y < 1 or y >= frame_h - 1:
                break

            if road_mask[y, x] > 0:
                best_len = float(length)
            else:
                # допускаем небольшой разрыв, если рядом ещё дорога
                patch = road_mask[max(0, y - 4):min(frame_h, y + 5), max(0, x - 4):min(frame_w, x + 5)]
                if patch.size == 0 or np.mean(patch > 0) < 0.12:
                    break
                best_len = float(length)

        return float(np.clip(best_len, self._min_zone_length_px, max_len))

    def _score_candidate_polygon(
        self,
        polygon: list[tuple[int, int]],
        car_detections: list[Detection],
        road_mask: np.ndarray,
        direction: np.ndarray,
        anchor_point: np.ndarray,
        inward_normal: np.ndarray,
        lane_width: int,
        frame_w: int,
        frame_h: int,
    ) -> float:
        score = 0.0
        poly_np = np.array(polygon, dtype=np.int32)

        # 1. Машины в зоне — главный критерий.
        for det in car_detections:
            bbox = det.bbox
            pts = [
                (0.50 * (bbox.x1 + bbox.x2), bbox.y2),
                (bbox.x1 + 0.30 * (bbox.x2 - bbox.x1), bbox.y2 - 2),
                (bbox.x1 + 0.70 * (bbox.x2 - bbox.x1), bbox.y2 - 2),
                (bbox.x1 + 0.50 * (bbox.x2 - bbox.x1), bbox.y1 + 0.82 * (bbox.y2 - bbox.y1)),
            ]
            hits = sum(1 for x, y in pts if cv2.pointPolygonTest(poly_np, (float(x), float(y)), False) >= 0)
            if hits > 0:
                score += 6.0 + 1.5 * hits

        # 2. Поддержка дорогой: проверяем осевую линию внутри полигона.
        center_support = 0.0
        total = 0
        direction = self._normalize_vec(direction)
        inward_normal = self._normalize_vec(inward_normal)

        edge_len = np.linalg.norm(np.asarray(polygon[1], dtype=np.float32) - np.asarray(polygon[0], dtype=np.float32))
        samples = max(10, int(edge_len / 18.0))
        for i in range(1, samples + 1):
            t = i / float(samples)
            pt = anchor_point + direction * (edge_len * t) + inward_normal * (0.52 * lane_width)
            x = int(round(pt[0]))
            y = int(round(pt[1]))
            if 0 <= x < frame_w and 0 <= y < frame_h:
                total += 1
                if road_mask[y, x] > 0:
                    center_support += 1.0
        if total > 0:
            score += 5.0 * (center_support / total)

        # 3. Лёгкий бонус за “длинную, но разумную” зону.
        area = abs(cv2.contourArea(poly_np.astype(np.float32)))
        score += min(area / max(frame_w * frame_h, 1), 0.18) * 4.0

        return float(score)

    def _score_scene_geometry(
        self,
        polygon: list[tuple[int, int]],
        side: str,
        direction: np.ndarray,
        road_mask: np.ndarray,
        car_detections: list[Detection],
        frame_w: int,
        frame_h: int,
        locked_dir: np.ndarray | None = None,
    ) -> float:
        confidence = 0.28

        moving_tracks = self._moving_track_count(frame_h)
        confidence += min(0.24, 0.06 * moving_tracks)

        if locked_dir is not None:
            alignment = max(0.0, float(np.dot(self._normalize_vec(direction), self._normalize_vec(locked_dir))))
            confidence += 0.16 * alignment

        mask_support = self._polygon_mask_support(polygon, road_mask)
        confidence += 0.22 * mask_support

        poly_np = np.asarray(polygon, dtype=np.int32)
        car_hits = 0
        for detection in car_detections:
            bbox = detection.bbox
            foot = (0.5 * (bbox.x1 + bbox.x2), bbox.y2)
            if cv2.pointPolygonTest(poly_np, (float(foot[0]), float(foot[1])), False) >= 0:
                car_hits += 1
        confidence += min(0.16, 0.04 * car_hits)

        x1, y1, x2, y2 = polygon_bbox(polygon)
        if x2 <= x1 or y2 <= y1:
            confidence *= 0.25
        elif x1 <= 0 or y1 <= 0 or x2 >= frame_w - 1 or y2 >= frame_h - 1:
            confidence -= 0.06

        if side not in ("left", "right"):
            confidence -= 0.08

        return float(np.clip(confidence, 0.0, 0.90))

    @staticmethod
    def _polygon_mask_support(polygon: list[tuple[int, int]], road_mask: np.ndarray) -> float:
        if road_mask.size == 0:
            return 0.0

        h, w = road_mask.shape[:2]
        poly_np = np.asarray(polygon, dtype=np.int32)
        x1, y1, x2, y2 = polygon_bbox(polygon)
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            return 0.0

        local_poly = poly_np.copy()
        local_poly[:, 0] -= x1
        local_poly[:, 1] -= y1
        poly_mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [local_poly], 1)
        area = int(np.count_nonzero(poly_mask))
        if area == 0:
            return 0.0

        road_crop = road_mask[y1:y2 + 1, x1:x2 + 1] > 0
        support = np.count_nonzero((poly_mask > 0) & road_crop) / float(area)
        return float(np.clip(support, 0.0, 1.0))

    # ---------- road geometry ----------

    def _collect_road_profile(self, road_mask: np.ndarray) -> list[tuple[int, int, int]]:
        height = road_mask.shape[0]
        profile: list[tuple[int, int, int]] = []
        for y in range(height - 1, -1, -4):
            edges = self._get_road_edges(road_mask, y, band=3)
            if edges is None:
                continue
            left, right = edges
            if right - left < 24:
                continue
            profile.append((y, left, right))
        profile.reverse()
        return profile

    def _estimate_anchor_on_road_edge(
        self,
        detection: Detection,
        profile: list[tuple[int, int, int]],
        frame_h: int,
        locked_side: str | None = None,
    ) -> tuple[int, str, int] | None:
        x1, _, x2, y2 = detection.bbox.to_int_tuple()
        sign_x = float((x1 + x2) / 2.0)

        search_top = max(0, y2 - 6)
        search_bottom = min(frame_h - 1, y2 + int(frame_h * self._anchor_search_height_ratio))

        candidate_rows = [(y, left, right) for y, left, right in profile if search_top <= y <= search_bottom]
        if not candidate_rows:
            candidate_rows = [(y, left, right) for y, left, right in profile if abs(y - y2) <= int(frame_h * 0.12)]
        if not candidate_rows:
            return None

        best_score = float("inf")
        best_anchor: tuple[int, str, int] | None = None
        preferred_y = min(search_bottom, y2 + int(frame_h * 0.04))

        for y, left, right in candidate_rows:
            candidates: list[tuple[str, int, float]] = []

            if locked_side in (None, "unknown", "left"):
                candidates.append(("left", left, abs(sign_x - left)))
            if locked_side in (None, "unknown", "right"):
                candidates.append(("right", right, abs(sign_x - right)))

            for side, outer_x, edge_dist in candidates:
                if edge_dist > self._anchor_search_margin_px:
                    continue
                y_penalty = 0.38 * abs(y - preferred_y)
                outside_bonus = 0.0
                if side == "right" and sign_x >= outer_x:
                    outside_bonus = -12.0
                if side == "left" and sign_x <= outer_x:
                    outside_bonus = -12.0

                score = edge_dist + y_penalty + outside_bonus
                if score < best_score:
                    best_score = score
                    best_anchor = (y, side, int(outer_x))

        return best_anchor

    def _estimate_local_edge_vectors(
        self,
        profile: list[tuple[int, int, int]],
        anchor_y: int,
        side: str,
        frame_w: int,
        frame_h: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        local_rows = [(y, left, right) for y, left, right in profile if abs(y - anchor_y) <= max(28, int(frame_h * 0.12))]
        if len(local_rows) < 3:
            return None, None, float(frame_w * 0.30)

        edge_points = []
        widths = []
        centers = []

        for y, left, right in local_rows:
            outer_x = right if side == "right" else left
            edge_points.append((float(outer_x), float(y)))
            widths.append(float(right - left))
            centers.append((0.5 * (left + right), float(y)))

        pts = np.asarray(edge_points, dtype=np.float32)
        center = np.mean(pts, axis=0, keepdims=True)
        pts_centered = pts - center

        if pts_centered.shape[0] < 2:
            return None, None, float(np.median(widths) if widths else frame_w * 0.30)

        cov = np.cov(pts_centered.T)
        vals, vecs = np.linalg.eigh(cov)
        tangent = vecs[:, int(np.argmax(vals))].astype(np.float32)
        tangent = self._normalize_vec(tangent)

        # normal в сторону дороги
        normal_a = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        normal_b = -normal_a

        road_center = np.mean(np.asarray(centers, dtype=np.float32), axis=0)
        test_pt_a = center[0] + normal_a * 20.0
        test_pt_b = center[0] + normal_b * 20.0

        dist_a = float(np.linalg.norm(test_pt_a - road_center))
        dist_b = float(np.linalg.norm(test_pt_b - road_center))
        inward_normal = normal_a if dist_a < dist_b else normal_b
        inward_normal = self._normalize_vec(inward_normal)

        road_width = float(np.median(widths)) if widths else frame_w * 0.30
        return tangent, inward_normal, road_width

    @staticmethod
    def _get_road_edges(
        road_mask: np.ndarray,
        y: int,
        band: int = 5,
    ) -> tuple[int, int] | None:
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

    # ---------- vehicles / motion ----------

    def _update_vehicle_observations(self, car_detections: list[Detection], frame_w: int, frame_h: int) -> None:
        for detection in car_detections:
            bbox = detection.bbox
            box_w = max(1.0, bbox.x2 - bbox.x1)
            box_h = max(1.0, bbox.y2 - bbox.y1)
            if box_w < frame_w * 0.035 or box_h < frame_h * 0.035:
                continue

            cx = 0.5 * (bbox.x1 + bbox.x2)
            foot_y = bbox.y2
            if foot_y < frame_h * 0.28:
                continue

            self._vehicle_observations.append((float(cx), float(foot_y), float(box_w), float(box_h)))
            if detection.track_id is not None:
                history = self._track_points.setdefault(detection.track_id, deque(maxlen=20))
                history.append((float(cx), float(foot_y)))

    def _moving_track_count(self, frame_h: int) -> int:
        count = 0
        for history in self._track_points.values():
            if len(history) < 5:
                continue
            first = np.asarray(history[0], dtype=np.float32)
            last = np.asarray(history[-1], dtype=np.float32)
            if float(np.linalg.norm(last - first)) >= max(10.0, frame_h * 0.018):
                count += 1
        return count

    # ---------- state helpers ----------

    def _find_sign_id(self, detection: Detection) -> int:
        center_x = (detection.bbox.x1 + detection.bbox.x2) / 2.0
        center_y = (detection.bbox.y1 + detection.bbox.y2) / 2.0

        best_id = None
        best_dist = 56.0

        for sign_id, state in self._states.items():
            if state["class_id"] != detection.class_id:
                continue
            existing_center_x = (state["source_bbox"].x1 + state["source_bbox"].x2) / 2.0
            existing_center_y = (state["source_bbox"].y1 + state["source_bbox"].y2) / 2.0

            distance = abs(center_x - existing_center_x) + 0.35 * abs(center_y - existing_center_y)
            if distance < best_dist:
                best_dist = distance
                best_id = sign_id

        if best_id is None:
            best_id = self._next_id
            self._next_id += 1

        return best_id

    @staticmethod
    def _average_polygons(buffer: list[list[tuple[int, int]]]) -> list[tuple[int, int]] | None:
        if not buffer:
            return None
        target_len = max(len(poly) for poly in buffer)
        valid = [poly for poly in buffer if len(poly) == target_len]
        if not valid:
            return None
        pts = np.asarray(valid, dtype=np.float32)
        mean_pts = np.mean(pts, axis=0)
        return [(int(round(x)), int(round(y))) for x, y in mean_pts]

    @staticmethod
    def _smooth_polygon(
        prev_polygon: list[tuple[int, int]],
        new_polygon: list[tuple[int, int]],
        alpha: float,
    ) -> list[tuple[int, int]]:
        if len(prev_polygon) != len(new_polygon):
            return new_polygon
        prev = np.asarray(prev_polygon, dtype=np.float32)
        curr = np.asarray(new_polygon, dtype=np.float32)
        smoothed = (1.0 - alpha) * prev + alpha * curr
        return [(int(round(x)), int(round(y))) for x, y in smoothed]

    @staticmethod
    def _polygon_bbox_iou(first: list[tuple[int, int]], second: list[tuple[int, int]]) -> float:
        ax1, ay1, ax2, ay2 = polygon_bbox(first)
        bx1, by1, bx2, by2 = polygon_bbox(second)

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h

        area_a = max(0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0, (bx2 - bx1) * (by2 - by1))
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0 else 0.0

    @staticmethod
    def _normalize_vec(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return np.array([0.0, -1.0], dtype=np.float32)
        return (vector / norm).astype(np.float32)

    @staticmethod
    def _naive_side(detection: Detection, frame_w: int) -> str:
        sign_cx = 0.5 * (detection.bbox.x1 + detection.bbox.x2)
        return "right" if sign_cx >= frame_w * 0.35 else "left"

    def _scene_geometry_polygon(
        self,
        detection: Detection,
        rule: SignRule | None,
        frame_w: int,
        frame_h: int,
        locked_side: str | None = None,
        locked_dir: np.ndarray | None = None,
    ) -> tuple[list[tuple[int, int]], str, np.ndarray]:
        x1, _, x2, y2 = detection.bbox.to_int_tuple()
        sign_cx = 0.5 * (x1 + x2)
        sign_h = max(1.0, detection.bbox.y2 - detection.bbox.y1)
        side = locked_side if locked_side in ("left", "right") else self._naive_side(detection, frame_w)

        vanishing_point = self._estimate_vanishing_point(frame_w, frame_h)
        if vanishing_point is None:
            vanishing_point = self._fallback_vanishing_point(side, frame_w, frame_h)

        anchor_y = float(np.clip(
            y2 + max(sign_h * 3.0, frame_h * 0.16),
            frame_h * 0.35,
            frame_h * 0.92,
        ))

        side_strength = abs(sign_cx / max(frame_w, 1) - 0.5) * 2.0
        side_strength = float(np.clip(side_strength, 0.0, 1.0))
        edge_offset = frame_w * (0.075 + 0.245 * side_strength)
        if side == "right":
            anchor_x = sign_cx - edge_offset
        else:
            anchor_x = sign_cx + edge_offset
        anchor_x = float(np.clip(anchor_x, frame_w * 0.03, frame_w * 0.97))

        anchor = np.array([anchor_x, anchor_y], dtype=np.float32)
        forward_direction = self._normalize_vec(vanishing_point - anchor)
        if locked_dir is not None:
            locked = self._normalize_vec(np.asarray(locked_dir, dtype=np.float32))
            if float(np.dot(forward_direction, locked)) < 0.35:
                forward_direction = self._normalize_vec(0.65 * forward_direction + 0.35 * locked)

        if forward_direction[1] > -0.04:
            forward_direction = self._normalize_vec(self._fallback_vanishing_point(side, frame_w, frame_h) - anchor)

        rule_direction = rule.direction if rule is not None else "forward"
        direction = forward_direction
        anchor_point = anchor
        length_multiplier = 1.0
        if rule_direction == "backward":
            direction = -forward_direction
        elif rule_direction == "both":
            anchor_point = anchor - forward_direction * max(self._min_zone_length_px * 0.75, frame_h * 0.12)
            direction = forward_direction
            length_multiplier = 1.35

        inward_normal = np.array([direction[1], -direction[0]], dtype=np.float32)
        if side == "left":
            inward_normal *= -1.0
        inward_normal = self._normalize_vec(inward_normal)

        near_width = self._scene_lane_width(frame_w, frame_h)
        if rule_direction == "backward":
            far_width = min(float(self._lane_max_width_px), near_width * 1.18)
        elif rule_direction == "both":
            far_width = max(float(self._lane_min_width_px), near_width * 0.70)
        else:
            far_width = max(float(self._lane_min_width_px), near_width * 0.36)

        if rule_direction == "backward":
            far_y = min(frame_h * 0.98, anchor_y + frame_h * 0.42)
            if direction[1] > 1e-3:
                length_px = (far_y - anchor_y) / abs(float(direction[1]))
            else:
                length_px = max(frame_w, frame_h) * 0.42
        else:
            far_y = max(frame_h * 0.14, min(anchor_y - frame_h * 0.34, vanishing_point[1] + frame_h * 0.04))
            if direction[1] < -1e-3:
                length_px = (anchor_y - far_y) / abs(float(direction[1]))
            else:
                length_px = max(frame_w, frame_h) * 0.42
        length_px *= length_multiplier
        if rule is not None and rule.distance_m is not None:
            diag = float(np.hypot(frame_w, frame_h))
            length_px = (rule.distance_m / 25.0) * (0.35 * diag)
        length_px = float(np.clip(length_px, self._min_zone_length_px, max(frame_w, frame_h) * 0.92))

        polygon = self._build_polygon(
            anchor_point=anchor_point,
            direction=direction,
            inward_normal=inward_normal,
            near_width=int(round(near_width)),
            far_width=int(round(far_width)),
            length_px=length_px,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        return polygon, side, direction

    def _estimate_vanishing_point(self, frame_w: int, frame_h: int) -> np.ndarray | None:
        lines: list[tuple[float, float, float]] = []
        for history in self._track_points.values():
            if len(history) < 5:
                continue
            first = np.asarray(history[0], dtype=np.float32)
            last = np.asarray(history[-1], dtype=np.float32)
            if float(np.linalg.norm(last - first)) < max(10.0, frame_h * 0.018):
                continue
            a = float(first[1] - last[1])
            b = float(last[0] - first[0])
            c = float(first[0] * last[1] - last[0] * first[1])
            norm = float(np.hypot(a, b))
            if norm > 1e-6:
                lines.append((a / norm, b / norm, c / norm))

        if len(lines) < 2:
            return None

        intersections: list[tuple[float, float]] = []
        for i, first in enumerate(lines):
            a1, b1, c1 = first
            for a2, b2, c2 in lines[i + 1:]:
                denom = a1 * b2 - a2 * b1
                if abs(denom) < 1e-4:
                    continue
                x = (b1 * c2 - b2 * c1) / denom
                y = (c1 * a2 - c2 * a1) / denom
                if -1.5 * frame_w <= x <= 2.5 * frame_w and -1.2 * frame_h <= y <= 1.1 * frame_h:
                    intersections.append((float(x), float(y)))

        if not intersections:
            return None

        pts = np.asarray(intersections, dtype=np.float32)
        return np.median(pts, axis=0).astype(np.float32)

    @staticmethod
    def _fallback_vanishing_point(side: str, frame_w: int, frame_h: int) -> np.ndarray:
        x = frame_w * (0.88 if side == "right" else 0.12)
        y = frame_h * 0.28
        return np.array([x, y], dtype=np.float32)

    def _scene_lane_width(self, frame_w: int, frame_h: int) -> float:
        widths = []
        for _, foot_y, box_w, box_h in self._vehicle_observations:
            if foot_y < frame_h * 0.30:
                continue
            widths.append(min(box_w, box_h * 1.28))

        if not widths:
            return float(np.clip(frame_w * 0.24, self._lane_min_width_px, self._lane_max_width_px))

        observed = float(np.percentile(np.asarray(widths, dtype=np.float32), 65))
        return float(np.clip(observed * 0.95, frame_w * 0.18, min(frame_w * 0.36, self._lane_max_width_px)))

    def _fallback_camera_polygon(
        self,
        detection: Detection,
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[int, int]]:
        x1, y1, x2, y2 = detection.bbox.to_int_tuple()
        sign_cx = 0.5 * (x1 + x2)
        sign_w = max(1.0, x2 - x1)
        sign_h = max(1.0, y2 - y1)

        side = self._naive_side(detection, frame_w)
        near_y = float(np.clip(y2 + 2.0, 0, frame_h - 1))
        lane_width = float(np.clip(frame_w * 0.22, self._lane_min_width_px, self._lane_max_width_px))

        # Упрощённый fallback — от знака влево/вправо и слегка вниз.
        if side == "right":
            direction = np.array([-1.0, 0.20], dtype=np.float32)
            outward_x = sign_cx - max(sign_w * 0.6, 12.0)
        else:
            direction = np.array([1.0, 0.20], dtype=np.float32)
            outward_x = sign_cx + max(sign_w * 0.6, 12.0)

        direction = self._normalize_vec(direction)
        inward_normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        if side == "right" and inward_normal[0] > 0:
            inward_normal *= -1.0
        if side == "left" and inward_normal[0] < 0:
            inward_normal *= -1.0
        inward_normal = self._normalize_vec(inward_normal)

        return self._build_polygon(
            anchor_point=np.array([outward_x, near_y], dtype=np.float32),
            direction=direction,
            inward_normal=inward_normal,
            near_width=int(lane_width),
            far_width=int(lane_width * 0.9),
            length_px=float(max(frame_w, frame_h) * 0.45),
            frame_w=frame_w,
            frame_h=frame_h,
        )
