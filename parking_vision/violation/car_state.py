from collections import deque
from typing import Any

from perception_types import Detection

from .geometry import bbox_tuple_iou
from .types import SignZone, ViolationRecord


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

    def _ensure_state(self, track_id: int, timestamp_ms: float | None) -> dict[str, Any]:
        if track_id not in self.track_states:
            self.track_states[track_id] = {
                "first_seen_time": timestamp_ms,
                "last_seen_time": timestamp_ms,
                "plate": "",
                "center_history": deque(maxlen=max(self.stop_frames_threshold, 2)),
                "is_parked": False,
                "zone_entry_time": None,
                "stop_start_time": None,
                "active_zone": None,
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
        zone: SignZone | None,
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

        is_violation = stop_duration >= zone.time_limit_s if zone.time_limit_s > 0.0 else stop_duration >= 0.0
        if not is_violation:
            return None

        return ViolationRecord(
            track_id=track_id,
            plate=state.get("plate", ""),
            sign_id=zone.sign_id,
            sign_label=zone.sign_label,
            status="violation",
            time_in_zone_s=(timestamp_ms - entry_time) / 1000.0,
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
