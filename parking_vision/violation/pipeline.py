from typing import Any

from perception_types import BoundingBox, Detection, FrameDetections

from parking_vision.car_tracking.detector import CarTracker
from parking_vision.plate_reading.reader import PlateReader
from parking_vision.sign_detection.detector import SignDetector

from .car_state import CarStateManager
from .geometry import bbox_intersection_area
from .rendering import annotate_pipeline_frame
from .types import PipelineFrameResult, ViolationRecord
from .zone_manager import SignZoneManager


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
    ) -> None:
        self.car_tracker = CarTracker(model_path=car_model_path)
        self.sign_detector = SignDetector(model_path=sign_model_path)
        self.plate_reader = PlateReader(model_path=plate_model_path)
        self.sign_interval_frames = max(1, int(sign_interval_frames))
        self.plate_interval_frames = max(1, int(plate_interval_frames))
        self._last_sign_frame_index = -10**9
        self._last_sign_detections: list[Detection] = []
        self._last_plate_frame_index = -10**9
        self._last_plate_result: Any = []
        self.sign_zone_manager = SignZoneManager(
            parking_time_limit_s=parking_time_limit_s,
            max_missing_frames=30,
        )
        self.car_state_manager = CarStateManager(
            stop_distance_threshold_px=stop_distance_threshold_px,
            stop_frames_threshold=stop_frames_threshold,
        )

    def _has_plate_candidate(self) -> bool:
        return any(
            state.get("active_zone") is not None and state.get("is_parked")
            for state in self.car_state_manager.track_states.values()
        )

    def _assign_plate_matches(self, plate_matches: dict[int, str]) -> None:
        for track_id, text in plate_matches.items():
            if track_id in self.car_state_manager.track_states and text:
                self.car_state_manager.track_states[track_id]["plate"] = text

    @staticmethod
    def _normalize_plate_detections(plate_result: Any) -> list[dict[str, Any]]:
        if isinstance(plate_result, FrameDetections):
            return [
                {
                    "bbox": [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2],
                    "text": det.metadata.get("plate_text", ""),
                }
                for det in plate_result.detections
            ]
        return plate_result if isinstance(plate_result, list) else []

    def _match_plates_to_cars(
        self,
        car_detections: list[Detection],
        plate_result: Any,
    ) -> dict[int, str]:
        matches: dict[int, str] = {}
        for plate in self._normalize_plate_detections(plate_result):
            plate_box = BoundingBox(*plate["bbox"])
            plate_area = max(0.0, (plate_box.x2 - plate_box.x1) * (plate_box.y2 - plate_box.y1))
            if plate_area <= 0:
                continue

            best_id = None
            best_score = 0.0
            for car in car_detections:
                if car.track_id is None:
                    continue
                intersection = bbox_intersection_area(plate_box, car.bbox)
                if intersection > 0:
                    score = intersection / plate_area
                    if score > best_score:
                        best_score = score
                        best_id = car.track_id

            if best_id is not None and best_score > 0.3 and plate.get("text"):
                matches[best_id] = plate["text"]

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

    def process_frame(
        self,
        frame: Any,
        frame_index: int,
        timestamp_ms: float | None,
    ) -> tuple[Any, list[ViolationRecord], PipelineFrameResult]:
        car_frame = self.car_tracker.process_frame(frame, frame_index, timestamp_ms)
        sign_detections = self._get_sign_detections(frame, frame_index, timestamp_ms)
        active_zones = self.sign_zone_manager.build_zones(sign_detections, frame)

        violations: list[ViolationRecord] = []
        active_ids: set[int] = set()
        for car in car_frame.detections:
            if car.track_id is None:
                continue
            active_ids.add(car.track_id)
            zone = self.sign_zone_manager.find_zone_for_car(car, active_zones)
            self.car_state_manager.update_track(car, timestamp_ms, zone, "")

        self.car_state_manager.cleanup(timestamp_ms, active_ids)

        plate_matches: dict[int, str] = {}
        if self._has_plate_candidate():
            plate_frame = self._get_plate_result(frame, frame_index)
            plate_matches = self._match_plates_to_cars(car_frame.detections, plate_frame)
            self._assign_plate_matches(plate_matches)

        for car in car_frame.detections:
            if car.track_id is None:
                continue
            state = self.car_state_manager.get_track_status(car.track_id)
            violation = self.car_state_manager.build_violation(car.track_id, state, timestamp_ms)
            if violation is not None:
                violations.append(violation)

        result = PipelineFrameResult(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            car_detections=car_frame.detections,
            sign_detections=sign_detections,
            plate_matches=plate_matches,
            active_zones=active_zones,
            active_violations=violations,
        )
        annotated_frame = annotate_pipeline_frame(frame, result, self.car_state_manager)
        return annotated_frame, violations, result
