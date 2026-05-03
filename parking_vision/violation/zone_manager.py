from typing import Any

import cv2
import numpy as np

from perception_types import BoundingBox, Detection
from road import RoadSegmenter

from .constants import END_SIGN_ID, PARKING_SIGN_IDS, PROHIBITORY_SIGN_IDS, SIGN_LABELS, SIGN_TIME_LIMITS_S
from .geometry import polygon_bbox
from .types import SignZone


class SignZoneManager:
    def __init__(
        self,
        parking_time_limit_s: float = 300.0,
        max_missing_frames: int = 30,
        warmup_frames: int = 30,
        hist_check_interval: int = 10,
        hist_threshold: float = 0.6,
    ) -> None:
        self.segmenter = RoadSegmenter(update_every_n_frames=1, downscale=0.5)
        self._states: dict[int, dict[str, Any]] = {}
        self._next_id = 1
        self._warmup_frames = max(1, warmup_frames)
        self._hist_check_interval = max(1, hist_check_interval)
        self._hist_threshold = hist_threshold
        self._max_missing_frames = max(1, max_missing_frames)
        self.time_limits_s = dict(SIGN_TIME_LIMITS_S)
        self.time_limits_s[1] = float(parking_time_limit_s)
        self.time_limits_s[2] = float(parking_time_limit_s)
        self.time_limits_s[3] = float(parking_time_limit_s)

    def _calc_raw_polygon(
        self,
        detection: Detection,
        road_mask: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[int, int]]:
        num_points_per_side = 15
        x1, _, x2, y2 = detection.bbox.to_int_tuple()
        sign_base_y = min(y2 + 10, frame_h - 1)

        y_coords, _ = np.where(road_mask > 0)
        if len(y_coords) < 100:
            return self._fallback_polygon(detection, frame_w, frame_h)

        road_top_y = np.min(y_coords)
        road_bottom_y = np.max(y_coords)
        start_y = min(sign_base_y, road_bottom_y)
        end_y = road_top_y
        if start_y <= end_y + 5:
            return self._fallback_polygon(detection, frame_w, frame_h)

        y_steps = np.linspace(start_y, end_y, num_points_per_side).astype(int)
        left_side: list[tuple[int, int]] = []
        right_side: list[tuple[int, int]] = []
        last_left = None
        last_right = None

        for y in y_steps:
            edges = self._get_road_edges(road_mask, y, band=5)
            if edges:
                left, right = edges
                last_left, last_right = left, right
            elif last_left is not None and last_right is not None:
                left, right = last_left, last_right
            else:
                left, right = max(0, x1 - 200), min(frame_w, x2 + 200)

            left_side.append((left, y))
            right_side.append((right, y))

        return left_side + right_side[::-1]

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

    @staticmethod
    def _fallback_polygon(detection: Detection, width: int, height: int) -> list[tuple[int, int]]:
        vp_x, vp_y = width / 2.0, height * 0.45
        x1, _, x2, _ = detection.bbox.to_int_tuple()
        sign_cx = (x1 + x2) / 2.0
        y_near = float(height)
        x_near_right = sign_cx + 250
        x_near_left = sign_cx - 250
        progress = 0.7
        x_far_right = x_near_right + (vp_x - x_near_right) * progress
        x_far_left = x_near_left + (vp_x - x_near_left) * progress
        y_far = y_near + (vp_y - y_near) * progress
        return [
            (int(round(x_near_left)), int(round(y_near))),
            (int(round(x_near_right)), int(round(y_near))),
            (int(round(x_far_right)), int(round(y_far))),
            (int(round(x_far_left)), int(round(y_far))),
        ]

    @staticmethod
    def _average_polygons(buffer: list[list[tuple[int, int]]]) -> list[tuple[int, int]] | None:
        if not buffer:
            return None

        target_len = max(len(polygon) for polygon in buffer)
        valid_polygons = [polygon for polygon in buffer if len(polygon) == target_len]
        if not valid_polygons:
            return None

        points = np.array(valid_polygons, dtype=np.float32)
        mean_points = np.mean(points, axis=0)
        return [(int(round(x)), int(round(y))) for x, y in mean_points]

    def _find_sign_id(self, detection: Detection) -> int:
        center_x = (detection.bbox.x1 + detection.bbox.x2) / 2.0
        best_id = None
        best_dist = 40.0

        for sign_id, state in self._states.items():
            if state["class_id"] != detection.class_id:
                continue
            existing_center_x = (state["source_bbox"].x1 + state["source_bbox"].x2) / 2.0
            distance = abs(center_x - existing_center_x)
            if distance < best_dist:
                best_dist = distance
                best_id = sign_id

        if best_id is None:
            best_id = self._next_id
            self._next_id += 1

        return best_id

    @staticmethod
    def _get_histogram(frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    def build_zones(self, sign_detections: list[Detection], frame: np.ndarray) -> list[SignZone]:
        height, width = frame.shape[:2]
        road_mask = self.segmenter.get_road_mask(frame)
        active_ids: set[int] = set()
        current_hist = None

        for detection in sign_detections:
            if detection.class_id == END_SIGN_ID:
                continue
            if detection.class_id not in PROHIBITORY_SIGN_IDS and detection.class_id not in PARKING_SIGN_IDS:
                continue

            sign_id = self._find_sign_id(detection)
            active_ids.add(sign_id)

            if sign_id not in self._states:
                self._states[sign_id] = {
                    "state": "WARMUP",
                    "polygon_buffer": [],
                    "frozen_polygon": None,
                    "ref_histogram": None,
                    "frame_counter": 0,
                    "source_bbox": detection.bbox,
                    "class_id": detection.class_id,
                    "sign_label": SIGN_LABELS.get(detection.class_id, str(detection.class_id)),
                    "missed_frames": 0,
                }

            state = self._states[sign_id]
            state["source_bbox"] = detection.bbox
            state["missed_frames"] = 0

            if state["state"] == "RESET":
                state["state"] = "WARMUP"
                state["polygon_buffer"] = []
                state["frozen_polygon"] = None
                state["ref_histogram"] = None
                state["frame_counter"] = 0

            if state["state"] == "WARMUP":
                raw_polygon = self._calc_raw_polygon(detection, road_mask, width, height)
                state["polygon_buffer"].append(raw_polygon)
                state["frozen_polygon"] = self._average_polygons(state["polygon_buffer"])
                state["frame_counter"] += 1

                if state["frame_counter"] >= self._warmup_frames:
                    state["state"] = "LOCKED"
                    state["frame_counter"] = 0
                    state["frozen_polygon"] = self._average_polygons(state["polygon_buffer"])

                if current_hist is None:
                    current_hist = self._get_histogram(frame)
                state["ref_histogram"] = current_hist.copy()

        zones: list[SignZone] = []
        keys_to_delete: list[int] = []

        for sign_id, state in self._states.items():
            if sign_id not in active_ids:
                state["missed_frames"] += 1
                if state["missed_frames"] > self._max_missing_frames:
                    keys_to_delete.append(sign_id)
                continue

            if state["state"] == "LOCKED":
                state["frame_counter"] += 1
                if state["frame_counter"] % self._hist_check_interval == 0:
                    if current_hist is None:
                        current_hist = self._get_histogram(frame)
                    if state["ref_histogram"] is not None:
                        similarity = cv2.compareHist(
                            state["ref_histogram"],
                            current_hist,
                            cv2.HISTCMP_CORREL,
                        )
                        if similarity < self._hist_threshold:
                            state["state"] = "RESET"
                            state["polygon_buffer"] = []
                            state["frozen_polygon"] = None
                            state["ref_histogram"] = None
                            state["frame_counter"] = 0
                            continue

            if state["state"] == "RESET":
                continue

            if state["frozen_polygon"] is not None:
                x1, y1, x2, y2 = polygon_bbox(state["frozen_polygon"])
                zones.append(
                    SignZone(
                        sign_id=state["class_id"],
                        sign_label=state["sign_label"],
                        polygon=state["frozen_polygon"],
                        source_bbox=state["source_bbox"],
                        time_limit_s=self.time_limits_s.get(state["class_id"], 0.0),
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
        car_x = (car_detection.bbox.x1 + car_detection.bbox.x2) / 2.0
        car_y = car_detection.bbox.y2
        for zone in zones:
            if zone.contains_point(car_x, car_y):
                return zone
        return None
