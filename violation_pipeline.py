import argparse
import csv
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import cv2

from detect_and_read_plate import PlateReader
from detect_and_track_cars import CarTracker
from perception_types import BoundingBox, Detection, FrameDetections, ensure_parent_dir
from sign_detect import SignDetector


PROHIBITORY_SIGN_IDS = {0, 1, 2, 3}
END_SIGN_ID = 4
PARKING_SIGN_IDS = {5, 11, 12, 13}

SIGN_LABELS = {
    0: "3.27", 1: "3.28", 2: "3.29", 3: "3.30", 4: "3.31", 5: "6.4",
    6: "8.17", 7: "8.21", 8: "8.22", 9: "8.23", 10: "8.24", 11: "8.6.2", 12: "8.6.4", 13: "8.8",
}

SIGN_TIME_LIMITS_S = {0: 0.0, 1: 300.0, 2: 300.0, 3: 300.0}

@dataclass(slots=True)
class SignZone:
    sign_id: int
    sign_label: str
    polygon: list[tuple[int, int]]  
    source_bbox: BoundingBox
    time_limit_s: float
    _bbox_x1: int = field(default=0, repr=False)
    _bbox_y1: int = field(default=0, repr=False)
    _bbox_x2: int = field(default=0, repr=False)
    _bbox_y2: int = field(default=0, repr=False)

    def contains_point(self, x: float, y: float) -> bool:
        if not (self._bbox_x1 <= x <= self._bbox_x2 and self._bbox_y1 <= y <= self._bbox_y2): return False
        return cv2.pointPolygonTest(np.array(self.polygon, dtype=np.int32), (x, y), False) >= 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_bbox"] = self.source_bbox.to_dict()
        for k in ["_bbox_x1", "_bbox_y1", "_bbox_x2", "_bbox_y2"]: payload.pop(k, None)
        return payload

@dataclass(slots=True)
class ViolationRecord:
    track_id: int
    plate: str
    sign_id: int
    sign_label: str
    status: str
    time_in_zone_s: float
    stopped_duration_s: float
    bbox: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = self.bbox.to_dict()
        return payload

@dataclass(slots=True)
class PipelineFrameResult:
    frame_index: int
    timestamp_ms: float | None
    car_detections: list[Detection]
    sign_detections: list[Detection]
    plate_matches: dict[int, str]
    active_zones: list[SignZone]
    active_violations: list[ViolationRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index, "timestamp_ms": self.timestamp_ms,
            "car_detections": [i.to_dict() for i in self.car_detections],
            "sign_detections": [i.to_dict() for i in self.sign_detections],
            "plate_matches": self.plate_matches,
            "active_zones": [z.to_dict() for z in self.active_zones],
            "active_violations": [v.to_dict() for v in self.active_violations],
        }

@dataclass(slots=True)
class VideoOutputConfig:
    annotated_video_path: Path
    violations_csv_path: Path
    jsonl_path: Path


class SignZoneManager:
    def __init__(self, time_limits_s: dict[int, float] | None = None) -> None:
        self._lane_memory = {"left": None, "right": None}
        self._zone_memory: dict[int, list[tuple[int, int]]] = {}
        self._vp_memory: tuple[int, int] | None = None
        self._right_border_memory: dict[int, tuple[float, float]] = {}
        self._debug_local_right: list[dict[str, Any]] = []
        self._time_limits_s = dict(SIGN_TIME_LIMITS_S if time_limits_s is None else time_limits_s)

    def _fit_line(self, segments: list[tuple[int, int, int, int]]):
        if not segments:
            return None
        xs = []
        ys = []
        for x1, y1, x2, y2 in segments:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        return np.polyfit(ys, xs, 1)

    def _smooth_line(self, prev, curr, alpha=0.92):
        if curr is None:
            return prev
        if prev is None:
            return curr
        return (
            alpha * prev[0] + (1.0 - alpha) * curr[0],
            alpha * prev[1] + (1.0 - alpha) * curr[1],
        )

    def _line_x_at_y(self, line, y: int) -> int:
        k, b = line
        return int(k * y + b)

    def _line_y_at_x(self, line, x: int) -> int | None:
        k, b = line
        if abs(k) < 1e-6:
            return None
        return int((x - b) / k)

    def _compute_zone_start_y(
        self,
        sign_cx: int,
        right_line,
        vp_y: int,
        frame_height: int,
    ) -> int | None:
        if right_line is None or abs(float(right_line[0])) < 1e-6:
            return None
        y_road_under_sign = int((sign_cx - right_line[1]) / right_line[0])
        if vp_y < y_road_under_sign < frame_height:
            return y_road_under_sign
        return None

    def _fit_local_right_border(
        self,
        frame: np.ndarray,
        sign_cx: int,
        sign_y2: int,
        frame_width: int,
        frame_height: int,
    ):
        # The curb/right road edge is usually left-below from the sign pole.
        roi_x1 = max(0, sign_cx - int(frame_width * 0.22))
        roi_x2 = min(frame_width, sign_cx + int(frame_width * 0.03))
        roi_y1 = max(int(frame_height * 0.40), sign_y2 + int(frame_height * 0.02))
        roi_y2 = min(frame_height, sign_y2 + int(frame_height * 0.50))
        if roi_x2 - roi_x1 < 12 or roi_y2 - roi_y1 < 20:
            return None, (roi_x1, roi_y1, roi_x2, roi_y2)

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 140)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=18,
            minLineLength=max(18, int((roi_y2 - roi_y1) * 0.22)),
            maxLineGap=18,
        )
        if lines is None:
            return None, (roi_x1, roi_y1, roi_x2, roi_y2)

        best_line = None
        best_score = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            gx1, gy1 = x1 + roi_x1, y1 + roi_y1
            gx2, gy2 = x2 + roi_x1, y2 + roi_y1
            dx = gx2 - gx1
            dy = gy2 - gy1
            if abs(dx) < 4 or abs(dy) < 12:
                continue
            slope = dy / float(dx)
            # In these scenes the curb edge goes left-down from the sign, so dy/dx is usually negative.
            if slope > -0.15:
                continue

            x_bottom = gx1 if gy1 > gy2 else gx2
            y_bottom = gy1 if gy1 > gy2 else gy2
            x_top = gx2 if gy1 > gy2 else gx1
            y_top = gy2 if gy1 > gy2 else gy1

            if x_bottom >= sign_cx:
                continue
            if x_top > sign_cx + int(frame_width * 0.02):
                continue

            length_score = abs(dy)
            top_proximity = abs(sign_cx - x_top)
            bottom_offset = abs((sign_cx - int(frame_width * 0.10)) - x_bottom)
            score = top_proximity * 2.0 + bottom_offset * 0.3 - length_score * 0.1

            if best_score is None or score < best_score:
                best_score = score
                best_line = (gx1, gy1, gx2, gy2)

        if best_line is None:
            return None, (roi_x1, roi_y1, roi_x2, roi_y2)
        return self._fit_line([best_line]), (roi_x1, roi_y1, roi_x2, roi_y2)

    def _intersect_lines(self, line_a, line_b):
        k1, b1 = line_a
        k2, b2 = line_b
        if abs(k1 - k2) < 1e-6:
            return None
        y = (b2 - b1) / (k1 - k2)
        x = k1 * y + b1
        return int(x), int(y)

    def _project_point_to_y(self, x0: int, y0: int, y_target: int, vp_x: int, vp_y: int) -> int:
        if y0 == vp_y:
            return x0
        t = (y_target - y0) / float(vp_y - y0)
        return int(x0 + (vp_x - x0) * t)

    def _clamp_x(self, x: int, frame_width: int) -> int:
        return max(0, min(frame_width - 1, int(x)))

    def _line_is_reasonable(
        self,
        line,
        y_near: int,
        y_far: int,
        frame_width: int,
        expected_side: str,
    ) -> bool:
        if line is None:
            return False
        x_near = self._line_x_at_y(line, y_near)
        x_far = self._line_x_at_y(line, y_far)
        if expected_side == "left":
            return x_near < frame_width * 0.65 and x_far < frame_width * 0.78 and x_near < x_far
        return x_near > frame_width * 0.15 and x_far > frame_width * 0.4 and x_near < x_far

    def _mix(self, a: int, b: int, ratio: float) -> int:
        return int(a + (b - a) * ratio)

    def _sign_key(self, det: Detection, frame_width: int, frame_height: int) -> int:
        cx = int((det.bbox.x1 + det.bbox.x2) * 0.5)
        cy = int((det.bbox.y1 + det.bbox.y2) * 0.5)
        x_bucket = max(0, min(99, int(cx / max(1, frame_width * 0.05))))
        y_bucket = max(0, min(99, int(cy / max(1, frame_height * 0.05))))
        return det.class_id * 10000 + x_bucket * 100 + y_bucket

    def build_zones(
        self,
        sign_detections: list[Detection],
        frame_width: int,
        frame_height: int,
        frame: np.ndarray,
    ) -> list[SignZone]:
        self._debug_local_right = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        mask = np.zeros_like(edges)
        mask[int(frame_height * 0.45):, :] = 255
        edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=80,
            minLineLength=80,
            maxLineGap=50,
        )

        left_segments: list[tuple[int, int, int, int]] = []
        right_segments: list[tuple[int, int, int, int]] = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                x_bottom = x1 if y1 > y2 else x2
                x_top = x2 if y1 > y2 else x1
                if abs(x_top - x_bottom) < frame_width * 0.02:
                    continue
                if x_bottom < frame_width * 0.55:
                    left_segments.append((x1, y1, x2, y2))
                else:
                    right_segments.append((x1, y1, x2, y2))

        current_left_line = self._fit_line(left_segments)
        current_right_line = self._fit_line(right_segments)
        left_line = self._smooth_line(self._lane_memory["left"], current_left_line, alpha=0.94)
        right_line = self._smooth_line(self._lane_memory["right"], current_right_line, alpha=0.94)
        self._lane_memory["left"] = left_line
        self._lane_memory["right"] = right_line

        vp_x = int(frame_width * 0.42)
        vp_y = int(frame_height * 0.34)
        if left_line is not None and right_line is not None:
            vp = self._intersect_lines(left_line, right_line)
            if vp is not None:
                vp_x = max(0, min(frame_width - 1, vp[0]))
                vp_y = max(0, min(frame_height - 1, vp[1]))

        if self._vp_memory is None:
            self._vp_memory = (vp_x, vp_y)
        else:
            self._vp_memory = (
                int(self._vp_memory[0] * 0.92 + vp_x * 0.08),
                int(self._vp_memory[1] * 0.92 + vp_y * 0.08),
            )

        vp_x, vp_y = self._vp_memory
        vp_y = max(int(frame_height * 0.22), min(int(frame_height * 0.45), vp_y))

        relevant = [
            detection
            for detection in sign_detections
            if detection.class_id in PROHIBITORY_SIGN_IDS or detection.class_id == END_SIGN_ID
        ]
        relevant.sort(key=lambda detection: detection.bbox.y2)

        zones: list[SignZone] = []

        for index, det in enumerate(relevant):
            if det.class_id not in PROHIBITORY_SIGN_IDS:
                continue

            x1, y1, x2, y2 = det.bbox.to_int_tuple()
            sign_cx = max(0, min(frame_width - 1, (x1 + x2) // 2))
            pole_margin = max(18, int(frame_width * 0.03))
            near_width = max(130, int(frame_width * 0.24))
            far_width = max(60, int(frame_width * 0.10))
            sign_key = self._sign_key(det, frame_width, frame_height)
            local_right_line, local_roi = self._fit_local_right_border(frame, sign_cx, y2, frame_width, frame_height)
            if local_right_line is not None:
                local_right_line = self._smooth_line(self._right_border_memory.get(sign_key), local_right_line, alpha=0.9)
                self._right_border_memory[sign_key] = local_right_line
            active_right_line = local_right_line if local_right_line is not None else right_line
            self._debug_local_right.append(
                {
                    "sign_bbox": (x1, y1, x2, y2),
                    "roi": local_roi,
                    "local_right_line": local_right_line,
                    "active_right_line": active_right_line,
                }
            )

            y_far = max(vp_y + 24, int(frame_height * 0.34))
            for next_det in relevant[index + 1:]:
                if next_det.bbox.y2 >= y2:
                    continue
                if next_det.class_id == END_SIGN_ID or next_det.class_id in PROHIBITORY_SIGN_IDS:
                    y_far = max(vp_y + 24, int(next_det.bbox.y2 + max(8, frame_height * 0.02)))
                    break

            y_start = self._compute_zone_start_y(sign_cx, active_right_line, vp_y, frame_height)
            if y_start is None:
                y_start = min(frame_height - 40, max(vp_y + 140, y2 + int(frame_height * 0.22)))
            else:
                y_start = int(y2 + (y_start - y2) * 0.82)
            if y_start - y_far < 60:
                y_start = min(frame_height - 30, y_far + 60)

            if active_right_line is not None and self._line_is_reasonable(active_right_line, y_start, y_far, frame_width, "right"):
                right_bottom_x = self._line_x_at_y(active_right_line, y_start)
            else:
                right_bottom_x = sign_cx - pole_margin
            right_bottom_x = min(right_bottom_x, sign_cx - pole_margin)
            right_bottom_x = self._clamp_x(right_bottom_x, frame_width)

            left_bottom_x = right_bottom_x - near_width
            left_bottom_x = self._clamp_x(left_bottom_x, frame_width)
            if right_bottom_x - left_bottom_x < near_width:
                left_bottom_x = self._clamp_x(right_bottom_x - near_width, frame_width)

            if active_right_line is not None and self._line_is_reasonable(active_right_line, y_start, y_far, frame_width, "right"):
                right_top_x = self._line_x_at_y(active_right_line, y_far)
            else:
                right_top_x = self._project_point_to_y(right_bottom_x, y_start, y_far, vp_x, vp_y)
            right_top_x = min(right_top_x, vp_x + int(frame_width * 0.18))
            left_top_x = right_top_x - far_width

            if right_top_x - left_top_x < far_width:
                center_top_x = (left_top_x + right_top_x) // 2
                left_top_x = center_top_x - far_width // 2
                right_top_x = center_top_x + far_width // 2

            polygon = [
                (self._clamp_x(left_bottom_x, frame_width), y_start),
                (self._clamp_x(right_bottom_x, frame_width), y_start),
                (self._clamp_x(right_top_x, frame_width), y_far),
                (self._clamp_x(left_top_x, frame_width), y_far),
            ]

            key = sign_key
            prev = self._zone_memory.get(key)
            if prev is not None and len(prev) == 4:
                polygon = [
                    (
                        int(px * 0.97 + x * 0.03),
                        int(py * 0.97 + y * 0.03),
                    )
                    for (x, y), (px, py) in zip(polygon, prev)
                ]
            self._zone_memory[key] = polygon

            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            zones.append(
                SignZone(
                    sign_id=det.class_id,
                    sign_label=SIGN_LABELS.get(det.class_id, det.class_name),
                    polygon=polygon,
                    source_bbox=det.bbox,
                    time_limit_s=self._time_limits_s.get(det.class_id, 0.0),
                    _bbox_x1=min(xs),
                    _bbox_y1=min(ys),
                    _bbox_x2=max(xs),
                    _bbox_y2=max(ys),
                )
            )

        return zones

    @staticmethod
    def find_zone_for_car(car_detection: Detection, zones: list[SignZone]) -> SignZone | None:
        car_x = (car_detection.bbox.x1 + car_detection.bbox.x2) / 2
        car_y = car_detection.bbox.y2
        for zone in zones:
            if zone.contains_point(car_x, car_y):
                return zone
        return None


class CarStateManager:
    def __init__(self, stop_distance_threshold_px: float = 12.0, stop_frames_threshold: int = 15, stale_track_timeout_ms: float = 5000.0) -> None:
        self.stop_distance_threshold_px = stop_distance_threshold_px
        self.stop_frames_threshold = stop_frames_threshold
        self.stale_track_timeout_ms = stale_track_timeout_ms
        self.track_states: dict[int, dict[str, Any]] = {}

    def _ensure_state(self, track_id: int, timestamp_ms: float | None) -> dict[str, Any]:
        if track_id not in self.track_states:
            self.track_states[track_id] = {
                "first_seen_time": timestamp_ms, "last_seen_time": timestamp_ms, "plate": "",
                "center_history": deque(maxlen=max(self.stop_frames_threshold, 2)),
                "is_parked": False, "zone_entry_time": None, "stop_start_time": None,
                "active_zone": None, "last_bbox": None,
            }
        return self.track_states[track_id]

    def _is_stopped(self, history: deque[tuple[float, float]]) -> bool:
        if len(history) < self.stop_frames_threshold: return False
        xs = [p[0] for p in history]; ys = [p[1] for p in history]
        return (max(xs) - min(xs) <= self.stop_distance_threshold_px and max(ys) - min(ys) <= self.stop_distance_threshold_px)

    def update_track(self, detection: Detection, timestamp_ms: float | None, zone: SignZone | None, plate_text: str) -> dict[str, Any]:
        if detection.track_id is None: raise ValueError("Car detection must have track_id")
        state = self._ensure_state(detection.track_id, timestamp_ms)
        center = ((detection.bbox.x1 + detection.bbox.x2) / 2.0, (detection.bbox.y1 + detection.bbox.y2) / 2.0)
        state["last_seen_time"] = timestamp_ms
        state["center_history"].append(center)
        state["last_bbox"] = detection.bbox
        if plate_text: state["plate"] = plate_text

        was_parked = state["is_parked"]
        state["is_parked"] = self._is_stopped(state["center_history"])
        if state["is_parked"] and not was_parked: state["stop_start_time"] = timestamp_ms
        if not state["is_parked"]: state["stop_start_time"] = None

        if zone is None:
            state["active_zone"] = None; state["zone_entry_time"] = None
        else:
            prev_zone = state["active_zone"]
            changed = prev_zone is None or prev_zone.sign_id != zone.sign_id or prev_zone.polygon != zone.polygon
            state["active_zone"] = zone
            if changed or state["zone_entry_time"] is None: state["zone_entry_time"] = timestamp_ms
        return state

    def build_violation(self, track_id: int, state: dict[str, Any], timestamp_ms: float | None) -> ViolationRecord | None:
        zone = state.get("active_zone")
        if not zone or not state.get("is_parked") or not timestamp_ms: return None
        entry_t = state.get("zone_entry_time")
        if not entry_t: return None
        
        t_zone = max(0.0, (timestamp_ms - entry_t) / 1000.0)
        stop_t = max(0.0, (timestamp_ms - state.get("stop_start_time")) / 1000.0) if state.get("stop_start_time") else 0.0
        active = stop_t > 0.0 if zone.time_limit_s == 0.0 else stop_t >= zone.time_limit_s
        if not active or not state.get("last_bbox"): return None

        return ViolationRecord(track_id=track_id, plate=state.get("plate", ""), sign_id=zone.sign_id,
                               sign_label=zone.sign_label, status="violation", time_in_zone_s=t_zone,
                               stopped_duration_s=stop_t, bbox=state["last_bbox"])

    def get_track_status(self, track_id: int) -> dict[str, Any]: return self.track_states.get(track_id, {})
    def cleanup(self, timestamp_ms: float | None, active_ids: set[int]) -> None:
        if not timestamp_ms: return
        stale = [tid for tid, s in self.track_states.items() if tid not in active_ids and (s.get("last_seen_time") is None or timestamp_ms - s["last_seen_time"] > self.stale_track_timeout_ms)]
        for tid in stale: self.track_states.pop(tid, None)


class ViolationPipeline:
    def __init__(
        self,
        car_model_path: str = "models/cars.pt",
        sign_model_path: str = "models/best.pt",
        plate_model_path: str = "models/plates.pt",
        stop_distance_threshold_px: float = 12.0,
        stop_frames_threshold: int = 15,
        parking_time_limit_s: float = 300.0,
    ) -> None:
        self.car_tracker = CarTracker(model_path=car_model_path)
        self.sign_detector = SignDetector(model_path=sign_model_path)
        self.plate_reader = PlateReader(model_path=plate_model_path)
        time_limits_s = dict(SIGN_TIME_LIMITS_S)
        time_limits_s[1] = float(parking_time_limit_s)
        time_limits_s[2] = float(parking_time_limit_s)
        time_limits_s[3] = float(parking_time_limit_s)
        self.sign_zone_manager = SignZoneManager(time_limits_s=time_limits_s)
        self.car_state_manager = CarStateManager(stop_distance_threshold_px=stop_distance_threshold_px, stop_frames_threshold=stop_frames_threshold)

    @staticmethod
    def _bbox_intersection_area(a: BoundingBox, b: BoundingBox) -> float:
        x1, y1, x2, y2 = max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2)
        return float((x2 - x1) * (y2 - y1)) if x2 > x1 and y2 > y1 else 0.0

    def _normalize_plate_detections(self, plate_result: Any) -> list[dict[str, Any]]:
        if isinstance(plate_result, FrameDetections):
            return [{"bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2], "text": d.metadata.get("plate_text", "")} for d in plate_result.detections]
        return plate_result if isinstance(plate_result, list) else []

    def _match_plates_to_cars(self, car_detections: list[Detection], plate_result: Any) -> dict[int, str]:
        matches = {}
        for plate in self._normalize_plate_detections(plate_result):
            plate_box = BoundingBox(*plate["bbox"]); p_area = max(0.0, (plate_box.x2-plate_box.x1)*(plate_box.y2-plate_box.y1))
            if p_area <= 0: continue
            best_id, best_sc = None, 0.0
            for car in car_detections:
                if car.track_id is None: continue
                inter = self._bbox_intersection_area(plate_box, car.bbox)
                if inter > 0: 
                    sc = inter / p_area
                    if sc > best_sc: best_sc, best_id = sc, car.track_id
            if best_id and best_sc > 0.3 and plate.get("text"): matches[best_id] = plate["text"]
        return matches

    def process_frame(self, frame: Any, frame_index: int, timestamp_ms: float | None) -> tuple[Any, list[ViolationRecord], PipelineFrameResult]:
        h, w = frame.shape[:2]
        car_frame = self.car_tracker.process_frame(frame, frame_index, timestamp_ms)
        sign_frame = self.sign_detector.process_frame(frame, frame_index, timestamp_ms)
        plate_frame = self.plate_reader.process_frame(frame)

        active_zones = self.sign_zone_manager.build_zones(sign_frame.detections, w, h, frame)
        plate_matches = self._match_plates_to_cars(car_frame.detections, plate_frame)

        violations, active_ids = [], set()
        for car in car_frame.detections:
            if car.track_id is None: continue
            active_ids.add(car.track_id)
            zone = self.sign_zone_manager.find_zone_for_car(car, active_zones)
            state = self.car_state_manager.update_track(car, timestamp_ms, zone, plate_matches.get(car.track_id, ""))
            viol = self.car_state_manager.build_violation(car.track_id, state, timestamp_ms)
            if viol: violations.append(viol)

        self.car_state_manager.cleanup(timestamp_ms, active_ids)
        res = PipelineFrameResult(frame_index, timestamp_ms, car_frame.detections, sign_frame.detections, plate_matches, active_zones, violations)
        return self.annotate_frame(frame, res), violations, res

    def annotate_frame(self, frame: Any, res: PipelineFrameResult) -> Any:
        ann, ov = frame.copy(), frame.copy()
        for z in res.active_zones:
            pts = np.array(z.polygon, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(ov, [pts], (0, 0, 200))
            cv2.putText(ov, f"Zone {z.sign_label}", (z.polygon[0][0]+5, z.polygon[0][1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        ann = cv2.addWeighted(ov, 0.2, ann, 0.8, 0)

        for item in self.sign_zone_manager._debug_local_right:
            roi_x1, roi_y1, roi_x2, roi_y2 = item["roi"]
            cv2.rectangle(ann, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 180, 0), 1)
            cv2.putText(ann, "ROI", (roi_x1, max(18, roi_y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1, cv2.LINE_AA)

            local_right_line = item.get("local_right_line")
            if local_right_line is not None:
                line_y1 = roi_y1
                line_y2 = roi_y2
                line_x1 = self.sign_zone_manager._line_x_at_y(local_right_line, line_y1)
                line_x2 = self.sign_zone_manager._line_x_at_y(local_right_line, line_y2)
                cv2.line(ann, (line_x1, line_y1), (line_x2, line_y2), (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(ann, "local_right_line", (line_x1 + 4, min(line_y2, line_y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        for s in res.sign_detections:
            x1,y1,x2,y2 = s.bbox.to_int_tuple()
            cv2.rectangle(ann, (x1,y1), (x2,y2), (80,220,80), 2)
            cv2.putText(ann, SIGN_LABELS.get(s.class_id, s.class_name), (x1, max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,220,80), 2, cv2.LINE_AA)

        v_map = {v.track_id: v for v in res.active_violations}
        for c in res.car_detections:
            if c.track_id is None: continue
            x1,y1,x2,y2 = c.bbox.to_int_tuple()
            st = self.car_state_manager.get_track_status(c.track_id)
            plate = st.get("plate") or res.plate_matches.get(c.track_id, "unknown")
            status = "Стоит" if st.get("is_parked") else "Движется"
            zone = st.get("active_zone")
            z_timer = 0.0
            if zone and res.timestamp_ms and st.get("zone_entry_time"): z_timer = max(0.0, (res.timestamp_ms - st["zone_entry_time"])/1000.0)
            
            col = (0, 255, 255); w_txt = ""
            viol = v_map.get(c.track_id)
            if viol: col, w_txt = (0, 0, 255), f"НАРУШЕНИЕ {viol.sign_label}"
            elif zone and st.get("is_parked"): col = (0, 165, 255)

            cv2.rectangle(ann, (x1,y1), (x2,y2), col, 2)
            lines = [f"ID:{c.track_id} {status}", f"Plate: {plate}"]
            if zone: lines.append(f"Zone {zone.sign_label}: {z_timer:.1f}s")
            if w_txt: lines.append(w_txt)
            
            cy = max(24, y1 - 8)
            for l in lines:
                cv2.putText(ann, l, (x1, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA); cy += 22
        return ann

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified parking violation pipeline.")
    p.add_argument("--source", required=True); p.add_argument("--output", default="outputs/violations_annotated.mp4")
    p.add_argument("--violations-csv", dest="violations_csv_path", default="outputs/violations.csv")
    p.add_argument("--jsonl", dest="jsonl_path", default="outputs/violation_pipeline.jsonl")
    p.add_argument("--car-model", default="models/cars.pt"); p.add_argument("--sign-model", default="models/signs.pt")
    p.add_argument("--plate-model", default="models/plates.pt"); p.add_argument("--stop-distance-threshold", type=float, default=12.0)
    p.add_argument("--parking-time-limit", type=float, default=300.0)
    p.add_argument("--stop-frames-threshold", type=int, default=15); p.add_argument("--show", action="store_true")
    return p.parse_args()

def resolve_source(v: str) -> int | str: return int(v) if v.isdigit() else v

def open_writer(p: Path, c: cv2.VideoCapture, f: Any) -> cv2.VideoWriter:
    fps = c.get(cv2.CAP_PROP_FPS) or 25.0; h, w = f.shape[:2]
    return cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

def save_jsonl(p: Path, items: list[PipelineFrameResult]) -> None:
    ensure_parent_dir(p)
    with p.open("w", encoding="utf-8") as f:
        for i in items: f.write(json.dumps(i.to_dict(), ensure_ascii=False) + "\n")

def run_pipeline(source: int | str, pipeline: ViolationPipeline, out: VideoOutputConfig, show: bool = False) -> list[PipelineFrameResult]:
    ensure_parent_dir(out.annotated_video_path); ensure_parent_dir(out.violations_csv_path); ensure_parent_dir(out.jsonl_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {source}")
    ok, fr = cap.read()
    if not ok: cap.release(); raise RuntimeError(f"Cannot read frame")
    
    writer = open_writer(out.annotated_video_path, cap, fr); results = []
    try:
        with out.violations_csv_path.open("w", newline="", encoding="utf-8") as csv_f:
            wr = csv.writer(csv_f)
            wr.writerow(["frame_index","timestamp_ms","track_id","plate","sign_id","sign_label","status","time_in_zone_s","stopped_duration_s","x1","y1","x2","y2"])
            idx = 0
            while True:
                frame = fr if idx == 0 else None
                if frame is None:
                    ok, frame = cap.read()
                    if not ok: break
                ts = cap.get(cv2.CAP_PROP_POS_MSEC); ts_v = None if ts < 0 else float(ts)
                ann_f, viols, res = pipeline.process_frame(frame, idx, ts_v)
                results.append(res); writer.write(ann_f)
                for v in viols:
                    wr.writerow([idx, "" if ts_v is None else f"{ts_v:.2f}", v.track_id, v.plate, v.sign_id, v.sign_label, v.status, f"{v.time_in_zone_s:.2f}", f"{v.stopped_duration_s:.2f}", int(v.bbox.x1), int(v.bbox.y1), int(v.bbox.x2), int(v.bbox.y2)])
                if show:
                    cv2.imshow("Violation Pipeline", ann_f)
                    if cv2.waitKey(1) & 0xFF == 27: break
                idx += 1
    finally:
        cap.release(); writer.release(); cv2.destroyAllWindows()
    save_jsonl(out.jsonl_path, results)
    return results

def main() -> None:
    a = parse_args()
    p = ViolationPipeline(
        a.car_model,
        a.sign_model,
        a.plate_model,
        a.stop_distance_threshold,
        a.stop_frames_threshold,
        a.parking_time_limit,
    )
    out = VideoOutputConfig(Path(a.output), Path(a.violations_csv_path), Path(a.jsonl_path))
    run_pipeline(resolve_source(a.source), p, out, a.show)
    print(f"Saved to: {out.annotated_video_path}, {out.violations_csv_path}, {out.jsonl_path}")

if __name__ == "__main__":
    main()
