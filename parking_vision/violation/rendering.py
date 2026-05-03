from typing import Any

import cv2
import numpy as np

from .constants import SIGN_LABELS
from .types import PipelineFrameResult


def annotate_pipeline_frame(
    frame: Any,
    result: PipelineFrameResult,
    car_state_manager: Any,
) -> Any:
    annotated = frame.copy()
    overlay = frame.copy()

    for zone in result.active_zones:
        points = np.array(zone.polygon, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [points], (0, 0, 200))
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
        plate = state.get("plate") or result.plate_matches.get(car_detection.track_id, "unknown")
        status = "Стоит" if state.get("is_parked") else "Движется"
        zone = state.get("active_zone")
        zone_timer = 0.0

        if zone and result.timestamp_ms is not None and state.get("zone_entry_time") is not None:
            zone_timer = max(0.0, (result.timestamp_ms - state["zone_entry_time"]) / 1000.0)

        color = (0, 255, 255)
        extra_label = ""
        violation = violations_by_track.get(car_detection.track_id)
        if violation is not None:
            color = (0, 0, 255)
            extra_label = f"НАРУШЕНИЕ {violation.sign_label}"
        elif zone is not None and state.get("is_parked"):
            color = (0, 165, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        lines = [f"ID:{car_detection.track_id} {status}", f"Plate: {plate}"]
        if zone is not None:
            lines.append(f"Zone {zone.sign_label}: {zone_timer:.1f}s")
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
