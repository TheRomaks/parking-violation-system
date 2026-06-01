from typing import Any

import cv2
import numpy as np

from .constants import SIGN_LABELS
from .types import PipelineFrameResult


def annotate_pipeline_frame(
    frame: Any,
    result: PipelineFrameResult,
    car_state_manager: Any,
    draw_zone_debug: bool = True,
) -> Any:
    annotated = frame.copy()
    overlay = frame.copy()

    if draw_zone_debug:
        for zone in result.active_zones:
            hard_road_mask = getattr(zone, "hard_road_mask", None)
            side_mask = getattr(zone, "side_mask", None)
            if hard_road_mask is not None:
                contours, _ = cv2.findContours((hard_road_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (80, 80, 80), 1)
            if side_mask is not None:
                contours, _ = cv2.findContours((side_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 180, 220), 1)
            if zone.zone_mask is not None:
                overlay[zone.zone_mask > 0] = (0, 0, 200)
                if hard_road_mask is not None and hard_road_mask.shape[:2] == zone.zone_mask.shape[:2]:
                    leak_px = int(np.count_nonzero((zone.zone_mask > 0) & (hard_road_mask == 0)))
                    if leak_px > 0:
                        cv2.putText(
                            overlay,
                            f"ZONE LEAK OUTSIDE ROAD: {leak_px} px",
                            (12, 34),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
            else:
                points = np.array(zone.polygon, np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [points], True, (0, 0, 200), 2)
            cv2.putText(
                overlay,
                f"Zone {zone.sign_label}",
                (zone.polygon[0][0] + 5, zone.polygon[0][1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        annotated = cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0)

        for zone in result.active_zones:
            metadata = getattr(zone, "metadata", {}) or {}
            raw_anchor = metadata.get("raw_anchor") or metadata.get("raw_projected_sign_ground_point")
            stable_anchor = metadata.get("stable_anchor") or metadata.get("projected_sign_ground_point")
            direction = metadata.get("stable_traffic_dir") or metadata.get("zone_direction_vec")
            try:
                if isinstance(raw_anchor, (list, tuple)) and len(raw_anchor) == 2:
                    cv2.circle(annotated, (int(raw_anchor[0]), int(raw_anchor[1])), 5, (255, 120, 0), 2, cv2.LINE_AA)
                if isinstance(stable_anchor, (list, tuple)) and len(stable_anchor) == 2:
                    stable_pt = (int(stable_anchor[0]), int(stable_anchor[1]))
                    cv2.circle(annotated, stable_pt, 6, (0, 220, 0), -1, cv2.LINE_AA)
                    if isinstance(direction, (list, tuple)) and len(direction) == 2:
                        end = (
                            int(stable_pt[0] + float(direction[0]) * 42.0),
                            int(stable_pt[1] + float(direction[1]) * 42.0),
                        )
                        cv2.arrowedLine(annotated, stable_pt, end, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
            except (TypeError, ValueError):
                pass

    for sign_detection in result.sign_detections:
        x1, y1, x2, y2 = sign_detection.bbox.to_int_tuple()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 220, 80), 2)
        cv2.putText(
            annotated,
            SIGN_LABELS.get(sign_detection.class_id, sign_detection.class_name),
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 220, 80),
            2,
            cv2.LINE_AA,
        )

    violations_by_track = {violation.track_id: violation for violation in result.active_violations}

    for car_detection in result.car_detections:
        if car_detection.track_id is None:
            continue

        x1, y1, x2, y2 = car_detection.bbox.to_int_tuple()
        state = car_state_manager.get_track_status(car_detection.track_id)
        plate = state.get("plate") or result.plate_matches.get(car_detection.track_id) or "unknown"
        status = "Stopped" if state.get("is_parked") else "Moving"
        zone = state.get("active_zone")
        assignment = state.get("zone_assignment")
        zone_timer = 0.0

        if zone and result.timestamp_ms is not None and state.get("zone_entry_time") is not None:
            zone_timer = max(0.0, (result.timestamp_ms - state["zone_entry_time"]) / 1000.0)

        color = (0, 255, 255)
        extra_label = ""
        violation = violations_by_track.get(car_detection.track_id)
        if violation is not None:
            color = (0, 0, 255)
            extra_label = f"Violation {violation.sign_label}"
        elif zone is not None and state.get("is_parked"):
            color = (0, 165, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        lines = [f"ID:{car_detection.track_id} {status}", f"Plate: {plate}"]
        if zone is not None:
            probability = getattr(assignment, "probability", None)
            if probability is None:
                lines.append(f"Zone {zone.sign_label}: {zone_timer:.1f}s")
            else:
                lines.append(f"Zone {zone.sign_label} p={probability:.2f}: {zone_timer:.1f}s")
        elif assignment is not None:
            lines.append(
                f"Zone {assignment.decision} {assignment.zone.sign_label} p={assignment.probability:.2f}"
            )
        if extra_label:
            lines.append(extra_label)

        current_y = max(24, y1 - 8)
        for line in lines:
            cv2.putText(
                annotated,
                line,
                (x1, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            current_y += 22

    return annotated
