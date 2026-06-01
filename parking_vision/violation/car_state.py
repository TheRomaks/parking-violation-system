from collections import deque
from typing import Any

from perception_types import Detection

from .geometry import bbox_tuple_iou
from .types import SignZone, ViolationRecord, ZoneAssignment


class CarStateManager:
    def __init__(
        self,
        stop_distance_threshold_px: float = 12.0,
        stop_frames_threshold: int = 15,
        stale_track_timeout_ms: float = 5000.0,
    ) -> None:
        self.stop_distance_threshold_px = stop_distance_threshold_px
        self.stop_frames_threshold = stop_frames_threshold
        self.stale_track_timeout_ms = stale_track_timeout_ms
        self.track_states: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _clamp_confidence(value: float | None) -> float:
        if value is None:
            return 0.0
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _plate_observation_score(
        cls,
        text: str,
        ocr_conf: float | None,
        detection_conf: float | None,
        valid: bool,
        bbox: list[float] | tuple[float, float, float, float] | None = None,
    ) -> float:
        if not text:
            return 0.0
        length_bonus = min(len(text), 9) / 9.0 * 0.15
        validity_bonus = 1.0 if valid else 0.0
        area_bonus = 0.0
        if bbox is not None and len(bbox) == 4:
            width = max(0.0, float(bbox[2]) - float(bbox[0]))
            height = max(0.0, float(bbox[3]) - float(bbox[1]))
            area_bonus = min(width * height / 4000.0, 1.0) * 0.35
        return (
            cls._clamp_confidence(ocr_conf) * 1.35
            + cls._clamp_confidence(detection_conf) * 0.25
            + validity_bonus
            + length_bonus
            + area_bonus
        )

    @staticmethod
    def _plate_candidate_score(candidate: dict[str, Any], timestamp_ms: float | None) -> float:
        count = int(candidate.get("count", 0))
        stability_bonus = min(max(count - 1, 0), 4) * 0.08
        score_sum = float(candidate.get("score_sum", 0.0))
        recent_bonus = 0.0
        last_seen = candidate.get("last_seen_time")
        if timestamp_ms is not None and last_seen is not None:
            age_ms = max(0.0, float(timestamp_ms) - float(last_seen))
            recent_bonus = max(0.0, 1.0 - age_ms / 2500.0) * 0.55
        return float(candidate.get("best_score", 0.0)) + stability_bonus + min(score_sum, 6.0) * 0.02 + recent_bonus

    def _ensure_state(self, track_id: int, timestamp_ms: float | None) -> dict[str, Any]:
        if track_id not in self.track_states:
            self.track_states[track_id] = {
                "first_seen_time": timestamp_ms,
                "last_seen_time": timestamp_ms,
                "plate": "",
                "plate_best_score": 0.0,
                "plate_candidates": {},
                "plate_bbox": None,
                "plate_detection_conf": 0.0,
                "plate_ocr_conf": 0.0,
                "center_history": deque(maxlen=max(self.stop_frames_threshold, 2)),
                "is_parked": False,
                "zone_entry_time": None,
                "stop_start_time": None,
                "active_zone": None,
                "zone_assignment": None,
                "last_bbox": None,
            }
        return self.track_states[track_id]

    def get_motion_vector(self, track_id: int) -> tuple[float, float] | None:
        state = self.track_states.get(track_id)
        if not state or len(state["center_history"]) < 5:
            return None
        points = list(state["center_history"])
        return points[-1][0] - points[0][0], points[-1][1] - points[0][1]

    def _is_stopped(self, history: deque[tuple[float, float]]) -> bool:
        if len(history) < self.stop_frames_threshold:
            return False
        xs = [point[0] for point in history]
        ys = [point[1] for point in history]
        return (
            max(xs) - min(xs) <= self.stop_distance_threshold_px
            and max(ys) - min(ys) <= self.stop_distance_threshold_px
        )

    def update_track(
        self,
        detection: Detection,
        timestamp_ms: float | None,
        assignment: ZoneAssignment | SignZone | None,
        plate_text: str,
    ) -> dict[str, Any]:
        if detection.track_id is None:
            raise ValueError("Car detection must have track_id")

        state = self._ensure_state(detection.track_id, timestamp_ms)
        center = (
            (detection.bbox.x1 + detection.bbox.x2) / 2.0,
            (detection.bbox.y1 + detection.bbox.y2) / 2.0,
        )
        state["last_seen_time"] = timestamp_ms
        state["center_history"].append(center)
        state["last_bbox"] = detection.bbox

        if plate_text:
            state["plate"] = plate_text

        if isinstance(assignment, SignZone):
            zone = assignment
            state["zone_assignment"] = None
        elif assignment is not None and assignment.applies:
            zone = assignment.zone
            state["zone_assignment"] = assignment
        else:
            zone = None
            state["zone_assignment"] = assignment

        was_parked = state["is_parked"]
        state["is_parked"] = self._is_stopped(state["center_history"])
        if state["is_parked"] and not was_parked:
            state["stop_start_time"] = timestamp_ms
        elif not state["is_parked"]:
            state["stop_start_time"] = None

        if zone is None:
            state["active_zone"] = None
            state["zone_entry_time"] = None
        else:
            prev_zone = state["active_zone"]
            zone_changed = False
            if prev_zone is None or prev_zone.sign_id != zone.sign_id:
                zone_changed = True
            else:
                if bbox_tuple_iou(prev_zone.bbox_tuple(), zone.bbox_tuple()) < 0.55:
                    zone_changed = True

            state["active_zone"] = zone
            if zone_changed or state["zone_entry_time"] is None:
                state["zone_entry_time"] = timestamp_ms

        return state

    def update_plate_candidate(
        self,
        track_id: int,
        text: str,
        ocr_conf: float | None = None,
        detection_conf: float | None = None,
        valid: bool = False,
        bbox: list[float] | tuple[float, float, float, float] | None = None,
        timestamp_ms: float | None = None,
    ) -> dict[str, Any]:
        state = self._ensure_state(track_id, timestamp_ms)

        if bbox is not None:
            state["plate_bbox"] = [float(value) for value in bbox]
        state["plate_detection_conf"] = max(
            float(state.get("plate_detection_conf", 0.0)),
            self._clamp_confidence(detection_conf),
        )

        text = str(text or "").strip()
        if not text:
            return state

        observation_score = self._plate_observation_score(text, ocr_conf, detection_conf, valid, bbox)
        candidates = state.setdefault("plate_candidates", {})
        candidate = candidates.setdefault(
            text,
            {
                "text": text,
                "count": 0,
                "score_sum": 0.0,
                "best_score": 0.0,
                "best_ocr_conf": 0.0,
                "best_detection_conf": 0.0,
                "valid": False,
                "bbox": None,
                "last_seen_time": None,
            },
        )
        candidate["count"] = int(candidate["count"]) + 1
        candidate["score_sum"] = float(candidate["score_sum"]) * 0.88 + observation_score
        candidate["valid"] = bool(candidate["valid"] or valid)
        candidate["last_seen_time"] = timestamp_ms
        if observation_score >= float(candidate["best_score"]):
            candidate["best_score"] = observation_score
            candidate["best_ocr_conf"] = self._clamp_confidence(ocr_conf)
            candidate["best_detection_conf"] = self._clamp_confidence(detection_conf)
            candidate["bbox"] = state.get("plate_bbox")

        best_text = ""
        best_score = -1.0
        for candidate_text, candidate_data in candidates.items():
            candidate_score = self._plate_candidate_score(candidate_data, timestamp_ms)
            if candidate_score > best_score:
                best_text = candidate_text
                best_score = candidate_score

        if best_text:
            best_candidate = candidates[best_text]
            state["plate"] = best_text
            state["plate_best_score"] = best_score
            state["plate_ocr_conf"] = float(best_candidate.get("best_ocr_conf", 0.0))
            state["plate_detection_conf"] = max(
                float(state.get("plate_detection_conf", 0.0)),
                float(best_candidate.get("best_detection_conf", 0.0)),
            )
            if best_candidate.get("bbox") is not None:
                state["plate_bbox"] = best_candidate["bbox"]

        return state

    def build_violation(
        self,
        track_id: int,
        state: dict[str, Any],
        timestamp_ms: float | None,
    ) -> ViolationRecord | None:
        zone = state.get("active_zone")
        bbox = state.get("last_bbox")
        if zone is None or bbox is None or timestamp_ms is None:
            return None
        if not state.get("is_parked"):
            return None

        entry_time = state.get("zone_entry_time")
        if entry_time is None:
            return None

        stop_duration = 0.0
        if state.get("stop_start_time") is not None:
            stop_duration = max(0.0, (timestamp_ms - state["stop_start_time"]) / 1000.0)

        time_in_zone_s = (timestamp_ms - entry_time) / 1000.0
        if zone.time_limit_s > 0.0:
            is_violation = min(stop_duration, time_in_zone_s) >= zone.time_limit_s
        else:
            is_violation = stop_duration >= 0.0
        if not is_violation:
            return None

        return ViolationRecord(
            track_id=track_id,
            plate=state.get("plate", ""),
            sign_id=zone.sign_id,
            sign_label=zone.sign_label,
            status="violation",
            time_in_zone_s=time_in_zone_s,
            stopped_duration_s=stop_duration,
            bbox=bbox,
        )

    def get_track_status(self, track_id: int) -> dict[str, Any]:
        return self.track_states.get(track_id, {})

    def cleanup(self, timestamp_ms: float | None, active_ids: set[int]) -> None:
        if timestamp_ms is None:
            return

        stale_tracks: list[int] = []
        for track_id, state in self.track_states.items():
            if track_id in active_ids:
                continue
            last_seen = state.get("last_seen_time")
            if last_seen is None or (timestamp_ms - last_seen) > self.stale_track_timeout_ms:
                stale_tracks.append(track_id)

        for track_id in stale_tracks:
            self.track_states.pop(track_id, None)
