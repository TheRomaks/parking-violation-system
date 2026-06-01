from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import cv2
import numpy as np

from perception_types import Detection
from road_seg_yolo import RoadSegmenter

from .constants import SIGN_TIME_LIMITS_S
from .geometry import polygon_bbox
from .road_zone_builder import (
    RoadZoneBuildResult,
    build_road_zone,
    choose_right_hand_traffic_dir,
    collect_road_profile,
    estimate_anchor_on_road_edge,
    normalize_mask,
    normalize_vec,
    split_side_mask,
)
from .sign_rules import (
    SignRule,
    build_rule,
    group_sign_stacks,
    is_zone_terminator,
    normalize_sign_label,
)
from .types import SignZone


@dataclass(slots=True)
class ZoneCandidate:
    polygon: list[tuple[int, int]]
    zone_mask: np.ndarray | None
    side: str
    direction_vec: np.ndarray
    confidence: float
    source: str
    metadata: dict[str, Any]


class SignZoneManager:
    """Stateful wrapper around deterministic road-mask zone construction.

    The geometry source is intentionally narrow: raw road segmentation mask,
    sign bbox, projection split line, and the sign rule. Vehicle detections are
    accepted only for API compatibility and are not used for zone geometry.
    """

    def __init__(
        self,
        parking_time_limit_s: float = 300.0,
        max_missing_frames: int = 20,
        warmup_frames: int = 3,
        use_road_mask_geometry: bool = True,
        sign_lock_confidence: float = 0.50,
        sign_presence_confidence: float = 0.2,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.segmenter = RoadSegmenter(update_every_n_frames=1)
        self._states: dict[int, dict[str, Any]] = {}
        self._next_id = 1
        self._max_missing_frames = max(1, int(max_missing_frames))
        self._use_road_mask_geometry = bool(use_road_mask_geometry)
        self._sign_lock_confidence = float(sign_lock_confidence)
        self._sign_presence_confidence = float(sign_presence_confidence)
        self._now_provider = now_provider or datetime.now
        self.last_road_mask: np.ndarray | None = None

        self.time_limits_s = dict(SIGN_TIME_LIMITS_S)
        self.time_limits_s[1] = float(parking_time_limit_s)
        self.time_limits_s[2] = float(parking_time_limit_s)
        self.time_limits_s[3] = float(parking_time_limit_s)

        self._max_anchor_jump_px = 90.0

        # Compatibility knobs used by older tests. They no longer drive scene
        # geometry or vehicle fallbacks.
        self._warmup_frames = max(1, int(warmup_frames))
        self._anchor_search_margin_px = 170

    def _reset_scene(self) -> None:
        self._states.clear()
        self._next_id = 1

    def build_zones(
        self,
        sign_detections: list[Detection],
        frame: np.ndarray,
        car_detections: list[Detection] | None = None,
    ) -> list[SignZone]:
        del car_detections
        height, width = frame.shape[:2]
        road_masks = self.segmenter.get_masks(frame)
        road_mask = self._prepare_road_geometry_mask(road_masks["road"])
        self.last_road_mask = road_mask

        if getattr(self.segmenter, "last_scene_cut", False):
            self._reset_scene()

        stacks = group_sign_stacks(sign_detections)
        terminators = [item for item in sign_detections if is_zone_terminator(item)]
        now = self._now_provider()
        observed_ids: set[int] = set()

        for stack in stacks:
            detection = stack.main
            rule = build_rule(stack, self.time_limits_s.get(detection.class_id, 300.0), now)
            if not rule.creates_violation_zone:
                continue

            sign_id = self._find_sign_id(detection)
            observed_ids.add(sign_id)
            state = self._states.setdefault(
                sign_id,
                {
                    "source_bbox": detection.bbox,
                    "class_id": detection.class_id,
                    "sign_label": normalize_sign_label(detection),
                    "last_confidence": 0.0,
                    "current_confidence": 0.0,
                    "locked": False,
                    "missed_frames": 0,
                    "side": "unknown",
                    "direction_vec": None,
                    "anchor": None,
                    "zone_mask": None,
                    "side_mask": None,
                    "polygon": None,
                    "zone_metadata": {},
                    "rule": rule,
                },
            )
            state["source_bbox"] = detection.bbox
            state["class_id"] = detection.class_id
            state["current_confidence"] = float(detection.confidence)
            state["last_confidence"] = max(float(state.get("last_confidence", 0.0)), float(detection.confidence))
            if detection.confidence >= self._sign_lock_confidence:
                state["locked"] = True
            rule = self._select_effective_rule(state, rule)
            state["sign_label"] = rule.sign_label
            state["rule"] = rule
            state["missed_frames"] = 0

            if not self._use_road_mask_geometry:
                continue

            locked_side = state.get("side") if state.get("side") in ("left", "right") else None
            locked_anchor = state.get("anchor")
            locked_dir = state.get("direction_vec")

            result = build_road_zone(
                detection=detection,
                road_mask=road_mask,
                rule=rule,
                terminators=terminators,
                locked_side=locked_side,
                locked_anchor=locked_anchor,
                locked_dir=locked_dir,
                max_anchor_jump_px=self._max_anchor_jump_px,
            )
            if result is None:
                continue

            self._update_state_from_result(state, result, width, height)

        active_zone_ids = self._active_zone_ids(observed_ids)
        zones = self._build_active_zones(active_zone_ids)
        self._cleanup_missing(observed_ids)
        return zones

    def _active_zone_ids(self, observed_ids: set[int]) -> set[int]:
        active_ids: set[int] = set()
        for sign_id in observed_ids:
            state = self._states.get(sign_id)
            if state is None:
                continue
            if float(state.get("current_confidence", 0.0)) < self._sign_presence_confidence:
                continue
            if state.get("polygon") is None or state.get("rule") is None:
                continue
            active_ids.add(sign_id)
        return active_ids

    @staticmethod
    def _rule_has_plate_constraints(rule: SignRule) -> bool:
        return bool(
            rule.plate_labels
            or rule.distance_m is not None
            or rule.direction != "forward"
            or rule.start_mode != "from_sign"
        )

    def _select_effective_rule(self, state: dict[str, Any], current_rule: SignRule) -> SignRule:
        previous_rule = state.get("rule")
        if not state.get("locked") or previous_rule is None:
            return current_rule
        if not isinstance(previous_rule, SignRule):
            return current_rule
        if self._rule_has_plate_constraints(previous_rule) and not self._rule_has_plate_constraints(current_rule):
            return previous_rule
        return current_rule

    def _update_state_from_result(
        self,
        state: dict[str, Any],
        result: RoadZoneBuildResult,
        frame_w: int,
        frame_h: int,
    ) -> None:
        del frame_w, frame_h

        if state.get("side") not in ("left", "right"):
            state["side"] = result.side
        elif state["side"] != result.side and result.metadata.get("anchor_confidence", 0.0) >= 0.85:
            state["side"] = result.side

        previous_anchor = state.get("anchor")
        if previous_anchor is None:
            state["anchor"] = result.projected_sign_ground_point.astype(np.float32)
        else:
            state["anchor"] = result.projected_sign_ground_point.astype(np.float32)

        previous_dir = state.get("direction_vec")
        if previous_dir is None:
            state["direction_vec"] = normalize_vec(result.traffic_dir.astype(np.float32))
        else:
            current = normalize_vec(result.traffic_dir.astype(np.float32))
            previous = normalize_vec(np.asarray(previous_dir, dtype=np.float32))
            if float(np.dot(current, previous)) < 0.0 and result.metadata.get("direction_confidence") == "low":
                current = previous
            state["direction_vec"] = normalize_vec(0.86 * previous + 0.14 * current)

        state["zone_mask"] = result.zone_mask
        state["side_mask"] = result.side_mask
        state["polygon"] = result.polygon
        state["zone_metadata"] = dict(result.metadata)

    def _build_active_zones(self, active_ids: set[int]) -> list[SignZone]:
        zones: list[SignZone] = []
        for sign_id, state in self._states.items():
            if sign_id not in active_ids:
                continue
            polygon = state.get("polygon")
            rule: SignRule | None = state.get("rule")
            if polygon is None or rule is None:
                continue

            zone_mask = state.get("zone_mask")
            if zone_mask is not None and np.any(zone_mask > 0):
                ys, xs = np.where(zone_mask > 0)
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
            else:
                x1, y1, x2, y2 = polygon_bbox(polygon)

            direction_vec = normalize_vec(np.asarray(state.get("direction_vec", [0.0, -1.0]), dtype=np.float32))
            metadata = {
                **dict(rule.metadata),
                **dict(state.get("zone_metadata", {})),
                "sign_locked": bool(state.get("locked", False)),
                "sign_missed_frames": int(state.get("missed_frames", 0)),
                "last_sign_confidence": round(float(state.get("last_confidence", 0.0)), 3),
                "current_sign_confidence": round(float(state.get("current_confidence", 0.0)), 3),
                "zone_start_point": dict(state.get("zone_metadata", {})).get(
                    "projected_sign_ground_point",
                    [float(polygon[0][0]), float(polygon[0][1])],
                ),
                "zone_direction_vec": [float(direction_vec[0]), float(direction_vec[1])],
                "rule_direction": rule.direction,
                "rule_start_mode": rule.start_mode,
            }

            zones.append(
                SignZone(
                    sign_id=sign_id,
                    sign_label=state["sign_label"],
                    polygon=polygon,
                    source_bbox=state["source_bbox"],
                    time_limit_s=rule.time_limit_s,
                    restriction=rule.restriction,
                    applies_now=rule.applies_now,
                    side=state.get("side", "unknown"),
                    direction=rule.direction,
                    plate_labels=list(rule.plate_labels),
                    metadata=metadata,
                    zone_mask=zone_mask,
                    side_mask=state.get("side_mask"),
                    hard_road_mask=self.last_road_mask,
                    _bbox_x1=x1,
                    _bbox_y1=y1,
                    _bbox_x2=x2,
                    _bbox_y2=y2,
                )
            )
        return zones

    def _cleanup_missing(self, active_ids: set[int]) -> None:
        keys_to_delete: list[int] = []
        for sign_id, state in self._states.items():
            if sign_id in active_ids:
                continue
            state["missed_frames"] += 1
            state["current_confidence"] = 0.0
            if state["missed_frames"] > self._max_missing_frames:
                keys_to_delete.append(sign_id)
        for sign_id in keys_to_delete:
            del self._states[sign_id]

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

    def _find_sign_id(self, detection: Detection) -> int:
        center_x = (detection.bbox.x1 + detection.bbox.x2) / 2.0
        center_y = (detection.bbox.y1 + detection.bbox.y2) / 2.0
        best_id = None
        best_dist = 56.0

        for sign_id, state in self._states.items():
            if state["class_id"] != detection.class_id:
                continue
            existing = state["source_bbox"]
            existing_center_x = (existing.x1 + existing.x2) / 2.0
            existing_center_y = (existing.y1 + existing.y2) / 2.0
            distance = abs(center_x - existing_center_x) + 0.35 * abs(center_y - existing_center_y)
            if distance < best_dist:
                best_dist = distance
                best_id = sign_id

        if best_id is None:
            best_id = self._next_id
            self._next_id += 1
        return best_id

    @staticmethod
    def _prepare_road_geometry_mask(
        road_mask: np.ndarray,
        car_detections: list[Detection] | None = None,
    ) -> np.ndarray:
        del car_detections
        return normalize_mask(road_mask)

    # Compatibility wrappers for focused geometry tests and old callers.
    @staticmethod
    def _collect_road_profile(road_mask: np.ndarray) -> list[tuple[int, int, int]]:
        return collect_road_profile(road_mask)

    def _estimate_anchor_on_road_edge(
        self,
        detection: Detection,
        profile: list[tuple[int, int, int]],
        frame_w: int = 0,
        frame_h: int = 0,
        locked_side: str | None = None,
    ) -> tuple[int, str, int] | None:
        if frame_w <= 0:
            frame_w = max((right for _, _, right in profile), default=1) + 1
        if frame_h <= 0:
            frame_h = max((y for y, _, _ in profile), default=1) + 1
        result = estimate_anchor_on_road_edge(
            detection=detection,
            profile=profile,
            frame_w=frame_w,
            frame_h=frame_h,
            locked_side=locked_side,
        )
        if result is None:
            return None
        y, side, x, _ = result
        return y, side, x

    @staticmethod
    def _preferred_rule_direction(
        tangent: np.ndarray,
        side: str,
        rule: SignRule | None,
        locked_dir: np.ndarray | None = None,
        inward_normal: np.ndarray | None = None,
    ) -> np.ndarray:
        if inward_normal is None:
            base = normalize_vec(np.asarray(tangent, dtype=np.float32))
            normal = np.asarray([-base[1], base[0]], dtype=np.float32)
            inward_normal = normal if side == "right" else -normal
        direction, _ = choose_right_hand_traffic_dir(
            tangent=np.asarray(tangent, dtype=np.float32),
            inward_normal=np.asarray(inward_normal, dtype=np.float32),
            side=side,
            locked_dir=locked_dir,
        )
        if rule is not None and rule.direction == "backward":
            return -direction
        return direction

    @staticmethod
    def _select_split_components(
        side_mask: np.ndarray,
        anchor_point: np.ndarray,
        split_line_dir: np.ndarray,
        direction: np.ndarray,
        mode: str,
        frame_w: int,
        frame_h: int,
        length_px: float | None = None,
    ) -> np.ndarray | None:
        del split_line_dir, frame_w, frame_h
        rule_direction = "backward" if mode == "backward" else "both" if mode == "both" else "forward"
        return split_side_mask(
            side_mask=side_mask,
            anchor_point=anchor_point,
            traffic_dir=direction,
            rule_direction=rule_direction,
            length_px=length_px,
        )

    @staticmethod
    def _mask_iou(first: np.ndarray | None, second: np.ndarray | None) -> float:
        if first is None or second is None or first.shape[:2] != second.shape[:2]:
            return 0.0
        a = first > 0
        b = second > 0
        union = int(np.count_nonzero(a | b))
        if union == 0:
            return 0.0
        return float(np.count_nonzero(a & b) / union)
