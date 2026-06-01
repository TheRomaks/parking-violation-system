from typing import Any

import cv2

from perception_types import BoundingBox, Detection, FrameDetections

from parking_vision.car_tracking.detector import CarTracker
from parking_vision.plate_reading.reader import PlateReader
from parking_vision.sign_detection.detector import SignDetector

from .car_state import CarStateManager
from .geometry import bbox_intersection_area
from .rendering import annotate_pipeline_frame
from .types import PipelineFrameResult, ViolationRecord, ZoneAssignment
from .vehicle_filter import VehicleCandidateFilter
from .zone_manager import SignZoneManager
from .zone_reasoner import ZoneReasoner


class ViolationPipeline:
    def __init__(
        self,
        car_model_path: str = "models/cars.pt",
        sign_model_path: str = "models/best.pt",
        plate_model_path: str = "models/plates.pt",
        stop_distance_threshold_px: float = 12.0,
        stop_frames_threshold: int = 15,
        parking_time_limit_s: float = 300.0,
        sign_interval_frames: int = 1,
        plate_interval_frames: int = 4,
        draw_zone_debug: bool = True,
    ) -> None:
        self.plate_reader = PlateReader(model_path=plate_model_path)
        self.car_tracker = CarTracker(model_path=car_model_path)
        self.sign_detector = SignDetector(model_path=sign_model_path)
        self.sign_interval_frames = max(1, int(sign_interval_frames))
        self.plate_interval_frames = max(1, int(plate_interval_frames))
        self.draw_zone_debug = draw_zone_debug
        self._last_sign_frame_index = -10**9
        self._last_sign_detections: list[Detection] = []
        self._last_plate_frame_index = -10**9
        self._last_plate_result: Any = []
        self.sign_zone_manager = SignZoneManager(
            parking_time_limit_s=parking_time_limit_s,
            max_missing_frames=30,
        )
        self.zone_reasoner = ZoneReasoner()
        self.car_state_manager = CarStateManager(
            stop_distance_threshold_px=stop_distance_threshold_px,
            stop_frames_threshold=stop_frames_threshold,
        )
        self.vehicle_filter = VehicleCandidateFilter()

    @staticmethod
    def _point_inside_bbox(x: float, y: float, bbox: BoundingBox, margin_px: float = 12.0) -> bool:
        return (
            (bbox.x1 - margin_px) <= x <= (bbox.x2 + margin_px)
            and (bbox.y1 - margin_px) <= y <= (bbox.y2 + margin_px)
        )

    def _assign_plate_matches(
        self,
        plate_matches: dict[int, dict[str, Any]],
        timestamp_ms: float | None,
    ) -> dict[int, str]:
        best_matches: dict[int, str] = {}
        for track_id, plate in plate_matches.items():
            if track_id not in self.car_state_manager.track_states:
                continue
            state = self.car_state_manager.update_plate_candidate(
                track_id=track_id,
                text=str(plate.get("text", "")),
                ocr_conf=float(plate.get("ocr_conf", 0.0)),
                detection_conf=float(plate.get("conf", 0.0)),
                valid=bool(plate.get("valid", False)),
                bbox=plate.get("bbox"),
                timestamp_ms=timestamp_ms,
            )
            best_matches[track_id] = state.get("plate", "")
        return best_matches

    @staticmethod
    def _normalize_plate_detections(plate_result: Any) -> list[dict[str, Any]]:
        if isinstance(plate_result, FrameDetections):
            return [
                {
                    "bbox": [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2],
                    "text": det.metadata.get("plate_text", ""),
                    "conf": det.confidence,
                    "ocr_conf": det.metadata.get("ocr_conf", 0.0),
                    "valid": det.metadata.get("valid", False),
                }
                for det in plate_result.detections
            ]
        return plate_result if isinstance(plate_result, list) else []

    def _match_plates_to_cars(
        self,
        car_detections: list[Detection],
        plate_result: Any,
    ) -> dict[int, dict[str, Any]]:
        matches: dict[int, dict[str, Any]] = {}
        for plate in self._normalize_plate_detections(plate_result):
            plate_box = BoundingBox(*plate["bbox"])
            plate_area = max(0.0, (plate_box.x2 - plate_box.x1) * (plate_box.y2 - plate_box.y1))
            if plate_area <= 0:
                continue

            plate_center_x = (plate_box.x1 + plate_box.x2) / 2.0
            plate_center_y = (plate_box.y1 + plate_box.y2) / 2.0

            best_id = None
            best_score = 0.0
            for car in car_detections:
                if car.track_id is None:
                    continue

                center_inside = self._point_inside_bbox(plate_center_x, plate_center_y, car.bbox)
                intersection = bbox_intersection_area(plate_box, car.bbox)
                score = 0.0
                if intersection > 0:
                    score = max(score, intersection / plate_area)
                if center_inside:
                    score = max(score, 0.75)

                if score > best_score:
                    best_score = score
                    best_id = car.track_id

            if best_id is not None and best_score > 0.15:
                previous = matches.get(best_id)
                plate_quality = (
                    best_score
                    + float(plate.get("ocr_conf", 0.0))
                    + float(plate.get("conf", 0.0)) * 0.25
                    + (1.0 if plate.get("valid") else 0.0)
                )
                if previous is None or plate_quality > float(previous.get("match_quality", 0.0)):
                    plate_match = dict(plate)
                    plate_match["match_quality"] = plate_quality
                    plate_match["match_score"] = best_score
                    matches[best_id] = plate_match

        return matches

    def _get_sign_detections(
        self,
        frame: Any,
        frame_index: int,
        timestamp_ms: float | None,
    ) -> list[Detection]:
        if (frame_index - self._last_sign_frame_index) >= self.sign_interval_frames:
            result = self.sign_detector.process_frame(frame, frame_index, timestamp_ms)
            self._last_sign_detections = result.detections
            self._last_sign_frame_index = frame_index
        return self._last_sign_detections

    def _get_plate_result(self, frame: Any, frame_index: int) -> Any:
        if (frame_index - self._last_plate_frame_index) >= self.plate_interval_frames:
            self._last_plate_result = self.plate_reader.process_frame(frame)
            self._last_plate_frame_index = frame_index
        return self._last_plate_result

    @staticmethod
    def _draw_projection_debug_overlay(frame: Any, zones: list[Any]) -> Any:
        """Draw the road split line used for sign-zone construction.

        The line is written by SignZoneManager into zone.metadata["projection_line"].
        It starts at the projected sign base on the segmented ground plane and
        runs from the sign base into the road; SignZoneManager cuts the sign-side
        road mask by this same boundary and selects the component allowed by the rule.
        """
        if frame is None:
            return frame

        for zone in zones:
            metadata = getattr(zone, "metadata", {}) or {}
            line = metadata.get("projection_line")
            if not isinstance(line, (list, tuple)) or len(line) != 2:
                continue
            try:
                p0 = (int(round(float(line[0][0]))), int(round(float(line[0][1]))))
                p1 = (int(round(float(line[1][0]))), int(round(float(line[1][1]))))
            except (TypeError, ValueError, IndexError):
                continue

            cv2.line(frame, p0, p1, (0, 220, 0), 3, cv2.LINE_AA)

            anchor = metadata.get("projected_sign_ground_point")
            if isinstance(anchor, (list, tuple)) and len(anchor) == 2:
                try:
                    a = (int(round(float(anchor[0]))), int(round(float(anchor[1]))))
                    cv2.circle(frame, a, 6, (0, 220, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame, a, 9, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(
                        frame,
                        "sign projection / road split",
                        (max(0, a[0] - 90), max(18, a[1] - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )
                except (TypeError, ValueError):
                    pass

        return frame

    def process_frame(
        self,
        frame: Any,
        frame_index: int,
        timestamp_ms: float | None,
    ) -> tuple[Any, list[ViolationRecord], PipelineFrameResult]:
        car_frame = self.car_tracker.process_frame(frame, frame_index, timestamp_ms)
        sign_detections = self._get_sign_detections(frame, frame_index, timestamp_ms)
        active_zones = self.sign_zone_manager.build_zones(sign_detections, frame, car_frame.detections)
        car_detections = self.vehicle_filter.filter(
            detections=car_frame.detections,
            frame_shape=frame.shape,
            road_mask=self.sign_zone_manager.last_road_mask,
            zones=active_zones,
        )
        if getattr(self.sign_zone_manager.segmenter, "last_scene_cut", False):
            self.zone_reasoner.reset()

        violations: list[ViolationRecord] = []
        zone_assignments: list[ZoneAssignment] = []
        active_ids: set[int] = set()
        for car in car_detections:
            if car.track_id is None:
                continue
            active_ids.add(car.track_id)
            track_state = self.car_state_manager.get_track_status(car.track_id)
            assignment = self.zone_reasoner.assign_car_to_zone(
                car=car,
                zones=active_zones,
                timestamp_ms=timestamp_ms,
                track_state=track_state,
            )
            if assignment is not None:
                zone_assignments.append(assignment)
            self.car_state_manager.update_track(car, timestamp_ms, assignment, "")

        self.car_state_manager.cleanup(timestamp_ms, active_ids)

        plate_frame = self._get_plate_result(frame, frame_index)
        plate_observations = self._match_plates_to_cars(car_detections, plate_frame)
        plate_matches = self._assign_plate_matches(plate_observations, timestamp_ms)

        for car in car_detections:
            if car.track_id is None:
                continue
            state = self.car_state_manager.get_track_status(car.track_id)
            violation = self.car_state_manager.build_violation(car.track_id, state, timestamp_ms)
            if violation is not None:
                violations.append(violation)

        result = PipelineFrameResult(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            car_detections=car_detections,
            sign_detections=sign_detections,
            plate_matches=plate_matches,
            active_zones=active_zones,
            active_violations=violations,
            zone_assignments=zone_assignments,
        )
        annotated_frame = annotate_pipeline_frame(
            frame,
            result,
            self.car_state_manager,
            draw_zone_debug=self.draw_zone_debug,
        )
        if self.draw_zone_debug:
            annotated_frame = self._draw_projection_debug_overlay(annotated_frame, active_zones)
        return annotated_frame, violations, result
