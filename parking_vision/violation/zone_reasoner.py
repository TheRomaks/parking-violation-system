from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from perception_types import Detection

from .types import SignZone, ZoneAssignment


class ZoneReasoner:
    """Probabilistic sign-to-vehicle membership layer.

    This keeps the current geometric zones as one evidence source, but exposes a
    calibrated-ish assignment instead of a hard polygon membership decision.
    """

    def __init__(
        self,
        apply_threshold: float = 0.62,
        uncertain_threshold: float = 0.40,
        membership_threshold: float = 0.40,
        boundary_tolerance_px: float = 8.0,
        temporal_alpha: float = 0.35,
        stale_after_ms: float = 5000.0,
    ) -> None:
        self.apply_threshold = float(apply_threshold)
        self.uncertain_threshold = float(uncertain_threshold)
        self.membership_threshold = float(membership_threshold)
        self.boundary_tolerance_px = float(boundary_tolerance_px)
        self.temporal_alpha = float(temporal_alpha)
        self.stale_after_ms = float(stale_after_ms)
        self._smoothed_scores: dict[tuple[int, int, str], float] = {}
        self._last_seen_ms: dict[tuple[int, int, str], float | None] = {}
        self._positive_streaks: dict[tuple[int, int, str], int] = defaultdict(int)

    def reset(self) -> None:
        self._smoothed_scores.clear()
        self._last_seen_ms.clear()
        self._positive_streaks.clear()

    def assign_car_to_zone(
        self,
        car: Detection,
        zones: list[SignZone],
        timestamp_ms: float | None,
        track_state: dict[str, Any] | None = None,
    ) -> ZoneAssignment | None:
        if car.track_id is None or not zones:
            return None

        self._cleanup(timestamp_ms)

        best: ZoneAssignment | None = None
        for zone in zones:
            assignment = self._score_assignment(car, zone, timestamp_ms, track_state or {})
            if best is None or assignment.probability > best.probability:
                best = assignment

        return best

    def _score_assignment(
        self,
        car: Detection,
        zone: SignZone,
        timestamp_ms: float | None,
        track_state: dict[str, Any],
    ) -> ZoneAssignment:
        assert car.track_id is not None

        key = self._assignment_key(car.track_id, zone)
        zone_source = str(zone.metadata.get("zone_source", ""))
        if zone_source.startswith("road_mask") or zone.metadata.get("geometry_version", "").startswith("road_mask"):
            return self._score_road_mask_assignment(car, zone, timestamp_ms, key)

        raw_reasons = self._raw_reasons(car, zone, track_state)
        raw_score = float(np.clip(sum(raw_reasons.values()), 0.0, 1.0))

        previous = self._smoothed_scores.get(key, raw_score)
        smoothed = (1.0 - self.temporal_alpha) * previous + self.temporal_alpha * raw_score

        if raw_score >= self.uncertain_threshold:
            self._positive_streaks[key] += 1
        else:
            self._positive_streaks[key] = 0

        streak_bonus = min(0.10, 0.025 * self._positive_streaks[key])
        probability = float(np.clip(smoothed + streak_bonus, 0.0, 1.0))

        self._smoothed_scores[key] = probability
        self._last_seen_ms[key] = timestamp_ms

        decision = "not_applies"
        if not zone.applies_now:
            probability = 0.0
            decision = "not_applies"
        elif probability >= self.apply_threshold:
            decision = "applies"
        elif probability >= self.uncertain_threshold:
            decision = "uncertain"

        reasons = {name: round(float(value), 4) for name, value in raw_reasons.items() if abs(value) > 1e-6}
        return ZoneAssignment(
            track_id=car.track_id,
            zone=zone,
            probability=round(probability, 4),
            decision=decision,
            reasons=reasons,
            metadata={
                "raw_score": round(raw_score, 4),
                "positive_streak": int(self._positive_streaks[key]),
                "zone_source": zone.metadata.get("zone_source", ""),
                "zone_confidence": zone.metadata.get("zone_confidence", 0.0),
                "start_relation": zone.metadata.get("start_relation", "unknown"),
                "signed_start_progress_px": zone.metadata.get("signed_start_progress_px"),
            },
        )

    def _score_road_mask_assignment(
        self,
        car: Detection,
        zone: SignZone,
        timestamp_ms: float | None,
        key: tuple[int, int, str],
    ) -> ZoneAssignment:
        membership_ratio = self.vehicle_zone_membership(car, zone.zone_mask)
        foot_point = self._vehicle_support_points(car)[0]
        distance_px = self._distance_to_zone_px(foot_point, zone)
        near_boundary = distance_px <= self.boundary_tolerance_px

        if not zone.applies_now:
            decision = "not_applies"
            probability = 0.0
        elif membership_ratio >= self.membership_threshold:
            decision = "applies"
            probability = float(np.clip(0.70 + 0.30 * membership_ratio, self.apply_threshold, 1.0))
        elif membership_ratio > 0.0 or near_boundary:
            decision = "uncertain"
            distance_score = float(np.clip(1.0 - distance_px / max(self.boundary_tolerance_px, 1.0), 0.0, 1.0))
            probability = min(self.apply_threshold - 0.01, max(self.uncertain_threshold, 0.22 + 0.30 * distance_score))
        else:
            decision = "not_applies"
            probability = 0.03

        if decision == "not_applies":
            self._positive_streaks[key] = 0
            self._smoothed_scores[key] = probability
        else:
            self._positive_streaks[key] += 1
            self._smoothed_scores[key] = probability
        self._last_seen_ms[key] = timestamp_ms

        return ZoneAssignment(
            track_id=car.track_id,
            zone=zone,
            probability=round(float(probability), 4),
            decision=decision,
            reasons={
                "zone_mask_membership": round(float(membership_ratio), 4),
                "zone_distance_px": round(float(distance_px), 3),
                "rule_applies_now": 1.0 if zone.applies_now else 0.0,
            },
            metadata={
                "membership_ratio": round(float(membership_ratio), 4),
                "zone_distance_px": round(float(distance_px), 3),
                "zone_source": zone.metadata.get("zone_source", ""),
                "zone_confidence": zone.metadata.get("zone_confidence", 0.0),
                "hard_mask": True,
            },
        )

    def _raw_reasons(
        self,
        car: Detection,
        zone: SignZone,
        track_state: dict[str, Any],
    ) -> dict[str, float]:
        foot_points = self._vehicle_support_points(car)
        zone_hits = sum(1 for point in foot_points if self._point_in_zone(point, zone))
        hit_ratio = zone_hits / float(len(foot_points))

        zone_source = str(zone.metadata.get("zone_source", ""))
        is_road_mask_zone = zone_source.startswith("road_mask")
        if is_road_mask_zone:
            start_score = 0.0
            start_metadata = {"start_relation": "mask_based"}
        else:
            start_score, start_metadata = self._start_boundary_score(foot_points[0], zone)
        distance_score = self._distance_score(foot_points[0], zone)
        zone_confidence = self._zone_confidence(zone)
        temporal_prior = self._temporal_prior(track_state, zone)

        reasons: dict[str, float] = {
            "rule_applies_now": 0.08 if zone.applies_now else -0.70,
            "start_boundary": start_score,
            "zone_membership": 0.62 * hit_ratio,
            "near_zone": 0.08 * distance_score,
            "zone_confidence": 0.10 * zone_confidence,
            "temporal_prior": 0.10 * temporal_prior,
        }

        if is_road_mask_zone and hit_ratio <= 0.0 and distance_score < 0.15:
            reasons["outside_zone_mask"] = -0.55

        if track_state.get("is_parked"):
            reasons["stopped_track"] = 0.06

        if zone.side != "unknown":
            reasons["known_side"] = 0.04

        if zone.metadata.get("road_mask_support") is not None:
            try:
                reasons["road_support"] = 0.12 * float(zone.metadata["road_mask_support"])
            except (TypeError, ValueError):
                pass

        if zone.direction in ("forward", "backward", "both"):
            reasons["rule_direction_known"] = 0.03

        zone.metadata.update(start_metadata)
        return reasons

    @staticmethod
    def _vehicle_support_points(car: Detection) -> list[tuple[float, float]]:
        bbox = car.bbox
        return [
            (0.50 * (bbox.x1 + bbox.x2), bbox.y2),
            (bbox.x1 + 0.30 * (bbox.x2 - bbox.x1), bbox.y2 - 2),
            (bbox.x1 + 0.50 * (bbox.x2 - bbox.x1), bbox.y2 - 4),
            (bbox.x1 + 0.70 * (bbox.x2 - bbox.x1), bbox.y2 - 2),
            (0.50 * (bbox.x1 + bbox.x2), bbox.y1 + 0.82 * (bbox.y2 - bbox.y1)),
        ]

    @staticmethod
    def vehicle_zone_membership(car: Detection, zone_mask: np.ndarray | None) -> float:
        if zone_mask is None or getattr(zone_mask, "size", 0) == 0:
            return 0.0

        h, w = zone_mask.shape[:2]
        bbox = car.bbox
        x1 = int(np.clip(np.floor(bbox.x1), 0, w - 1))
        x2 = int(np.clip(np.ceil(bbox.x2), 0, w - 1))
        y1 = int(np.clip(np.floor(bbox.y1 + 0.62 * (bbox.y2 - bbox.y1)), 0, h - 1))
        y2 = int(np.clip(np.ceil(bbox.y2), 0, h - 1))
        footprint_ratio = 0.0
        if x2 > x1 and y2 > y1:
            footprint = zone_mask[y1:y2 + 1, x1:x2 + 1] > 0
            footprint_ratio = float(np.count_nonzero(footprint) / max(1, footprint.size))

        support_points = ZoneReasoner._vehicle_support_points(car)
        hits = 0
        for x, y in support_points:
            px = int(round(float(x)))
            py = int(round(float(y)))
            if 0 <= px < w and 0 <= py < h and zone_mask[py, px] > 0:
                hits += 1
        point_ratio = hits / float(len(support_points))
        return float(np.clip(max(point_ratio, footprint_ratio), 0.0, 1.0))

    @staticmethod
    def vehicle_in_zone(car: Detection, zone_mask: np.ndarray | None) -> float:
        return ZoneReasoner.vehicle_zone_membership(car, zone_mask)

    @staticmethod
    def _distance_to_zone_px(point: tuple[float, float], zone: SignZone) -> float:
        transform = getattr(zone, "zone_distance_transform", None)
        if transform is not None and getattr(transform, "size", 0) > 0:
            h, w = transform.shape[:2]
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if 0 <= x < w and 0 <= y < h:
                return float(transform[y, x])
            return float("inf")
        score = ZoneReasoner._distance_score(point, zone)
        if score >= 1.0:
            return 0.0
        x1, y1, x2, y2 = zone.bbox_tuple()
        scale = max(48.0, 0.20 * float(max(x2 - x1, y2 - y1, 1)))
        return float((1.0 - score) * scale)

    @staticmethod
    def _point_in_zone(point: tuple[float, float], zone: SignZone) -> bool:
        zone_mask = getattr(zone, "zone_mask", None)
        if zone_mask is not None and getattr(zone_mask, "size", 0) > 0:
            h, w = zone_mask.shape[:2]
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if x < 0 or x >= w or y < 0 or y >= h:
                return False
            if zone_mask[y, x] > 0:
                return True
            # Small tolerance for detector jitter at the tire/road contact point.
            patch = zone_mask[max(0, y - 2):min(h, y + 3), max(0, x - 2):min(w, x + 3)]
            return bool(patch.size and np.any(patch > 0))
        return bool(zone.contains_point(float(point[0]), float(point[1])))

    @staticmethod
    def _distance_score(point: tuple[float, float], zone: SignZone) -> float:
        zone_mask = getattr(zone, "zone_mask", None)
        if zone_mask is not None and getattr(zone_mask, "size", 0) > 0:
            h, w = zone_mask.shape[:2]
            px = int(round(float(point[0])))
            py = int(round(float(point[1])))
            if 0 <= px < w and 0 <= py < h and zone_mask[py, px] > 0:
                return 1.0

            x1, y1, x2, y2 = zone.bbox_tuple()
            scale = max(48.0, 0.20 * float(max(x2 - x1, y2 - y1, 1)))
            margin = int(round(scale))
            rx1 = max(0, min(px, x1) - margin)
            ry1 = max(0, min(py, y1) - margin)
            rx2 = min(w - 1, max(px, x2) + margin)
            ry2 = min(h - 1, max(py, y2) + margin)
            if rx2 <= rx1 or ry2 <= ry1:
                return 0.0

            local = zone_mask[ry1:ry2 + 1, rx1:rx2 + 1] > 0
            if not np.any(local):
                return 0.0
            lx = px - rx1
            ly = py - ry1
            if 0 <= lx < local.shape[1] and 0 <= ly < local.shape[0] and local[ly, lx]:
                return 1.0

            inverse = (~local).astype(np.uint8)
            dist = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)
            if 0 <= lx < dist.shape[1] and 0 <= ly < dist.shape[0]:
                return float(np.clip(1.0 - float(dist[ly, lx]) / scale, 0.0, 1.0))
            return 0.0

        polygon = np.asarray(zone.polygon, dtype=np.int32)
        signed_distance = cv2.pointPolygonTest(polygon, point, True)
        if signed_distance >= 0.0:
            return 1.0

        x1, y1, x2, y2 = zone.bbox_tuple()
        scale = max(48.0, 0.25 * float(max(x2 - x1, y2 - y1, 1)))
        return float(np.clip(1.0 + signed_distance / scale, 0.0, 1.0))

    @staticmethod
    def _start_boundary_score(
        point: tuple[float, float],
        zone: SignZone,
    ) -> tuple[float, dict[str, float | str]]:
        if zone.direction == "both":
            return 0.08, {"start_relation": "both"}

        start = ZoneReasoner._metadata_point(zone, "zone_start_point")
        direction = ZoneReasoner._metadata_vector(zone, "zone_direction_vec")
        if start is None:
            start = np.asarray(zone.polygon[0], dtype=np.float32)
        if direction is None and len(zone.polygon) >= 2:
            direction = np.asarray(zone.polygon[1], dtype=np.float32) - np.asarray(zone.polygon[0], dtype=np.float32)
        if direction is None:
            return 0.0, {"start_relation": "unknown"}

        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return 0.0, {"start_relation": "unknown"}

        direction = direction / norm
        progress_px = float(np.dot(np.asarray(point, dtype=np.float32) - start, direction))

        x1, y1, x2, y2 = zone.bbox_tuple()
        tolerance_px = max(24.0, 0.10 * float(max(x2 - x1, y2 - y1, 1)))
        if progress_px < -tolerance_px:
            return -0.90, {
                "start_relation": "before_start",
                "signed_start_progress_px": round(progress_px, 3),
            }
        if progress_px < tolerance_px:
            return 0.02, {
                "start_relation": "near_start",
                "signed_start_progress_px": round(progress_px, 3),
            }
        return 0.18, {
            "start_relation": "after_start",
            "signed_start_progress_px": round(progress_px, 3),
        }

    @staticmethod
    def _metadata_point(zone: SignZone, key: str) -> np.ndarray | None:
        value = zone.metadata.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            return np.asarray([float(value[0]), float(value[1])], dtype=np.float32)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _metadata_vector(zone: SignZone, key: str) -> np.ndarray | None:
        return ZoneReasoner._metadata_point(zone, key)

    @staticmethod
    def _zone_confidence(zone: SignZone) -> float:
        value = zone.metadata.get("zone_confidence", 0.0)
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _temporal_prior(track_state: dict[str, Any], zone: SignZone) -> float:
        previous_zone = track_state.get("active_zone")
        if previous_zone is None:
            previous_assignment = track_state.get("zone_assignment")
            if previous_assignment is not None:
                previous_zone = getattr(previous_assignment, "zone", None)
        if previous_zone is None:
            return 0.0
        if previous_zone.sign_id != zone.sign_id:
            return 0.0
        return 1.0

    @staticmethod
    def _assignment_key(track_id: int, zone: SignZone) -> tuple[int, int, str]:
        return track_id, zone.sign_id, zone.sign_label

    def _cleanup(self, timestamp_ms: float | None) -> None:
        if timestamp_ms is None:
            return
        stale_keys = [
            key
            for key, last_seen in self._last_seen_ms.items()
            if last_seen is None or timestamp_ms - last_seen > self.stale_after_ms
        ]
        for key in stale_keys:
            self._last_seen_ms.pop(key, None)
            self._smoothed_scores.pop(key, None)
            self._positive_streaks.pop(key, None)