from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Callable

from perception_types import Detection

from .constants import (
    EVEN_DATE_SIGN_LABELS,
    EXCEPTION_PLATE_LABELS,
    NO_PARKING_LABELS,
    NO_STOPPING_LABELS,
    ODD_DATE_SIGN_LABELS,
    PROHIBITORY_SIGN_IDS,
    PROHIBITORY_SIGN_LABELS,
    SIGN_LABELS,
    SIGN_TIME_LIMITS_S,
    ZONE_DIRECTION_PLATE_LABELS,
    ZONE_DISTANCE_PLATE_LABELS,
    ZONE_END_PLATE_LABELS,
    ZONE_END_SIGN_LABELS,
    ZONE_INSIDE_PLATE_LABELS,
)


LabelProvider = Callable[[], datetime]


@dataclass(slots=True)
class SignRule:
    sign_id: int
    sign_label: str
    restriction: str
    time_limit_s: float
    applies_now: bool
    start_mode: str = "from_sign"
    direction: str = "forward"
    distance_m: float | None = None
    plate_labels: list[str] = field(default_factory=list)
    exception_labels: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def creates_violation_zone(self) -> bool:
        return self.applies_now


@dataclass(slots=True)
class SignStack:
    main: Detection
    label: str
    plates: list[Detection] = field(default_factory=list)

    @property
    def plate_labels(self) -> list[str]:
        return [normalize_sign_label(plate) for plate in self.plates]


def normalize_sign_label(detection: Detection) -> str:
    metadata_label = detection.metadata.get("sign_label") or detection.metadata.get("label")
    candidates = [
        SIGN_LABELS.get(detection.class_id),
        str(metadata_label) if metadata_label else "",
        detection.class_name,
    ]
    for value in candidates:
        normalized = _extract_sign_label(value)
        if normalized:
            return normalized
    return str(detection.class_id)


def _extract_sign_label(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace(",", ".")
    text = re.sub(r"(?<=\d)[_-](?=\d)", ".", text)
    match = re.search(r"\b\d(?:\.\d){1,2}\b", text)
    return match.group(0) if match else text


def is_prohibitory_sign(detection: Detection) -> bool:
    label = normalize_sign_label(detection)
    return detection.class_id in PROHIBITORY_SIGN_IDS or label in PROHIBITORY_SIGN_LABELS


def is_zone_terminator(detection: Detection) -> bool:
    label = normalize_sign_label(detection)
    return label in ZONE_END_SIGN_LABELS


def is_supplementary_plate(detection: Detection) -> bool:
    return normalize_sign_label(detection).startswith("8.")


def group_sign_stacks(detections: list[Detection]) -> list[SignStack]:
    main_signs = [item for item in detections if is_prohibitory_sign(item)]
    plates = [item for item in detections if is_supplementary_plate(item)]
    stacks: list[SignStack] = []

    for main in main_signs:
        mx1, my1, mx2, my2 = main.bbox.to_int_tuple()
        main_w = max(1.0, float(mx2 - mx1))
        main_h = max(1.0, float(my2 - my1))
        main_cx = 0.5 * (mx1 + mx2)

        attached: list[Detection] = []
        for plate in plates:
            px1, py1, px2, py2 = plate.bbox.to_int_tuple()
            plate_cx = 0.5 * (px1 + px2)
            plate_cy = 0.5 * (py1 + py2)
            x_ok = abs(plate_cx - main_cx) <= max(main_w * 1.25, 42.0)
            y_ok = (my1 - 0.35 * main_h) <= plate_cy <= (my2 + 3.25 * main_h)
            if x_ok and y_ok:
                attached.append(plate)

        attached.sort(key=lambda item: (item.bbox.y1, item.bbox.x1))
        stacks.append(SignStack(main=main, label=normalize_sign_label(main), plates=attached))

    stacks.sort(key=lambda stack: (stack.main.bbox.y1, stack.main.bbox.x1))
    return stacks


def build_rule(
    stack: SignStack,
    parking_time_limit_s: float,
    now: datetime | None = None,
) -> SignRule:
    now = now or datetime.now()
    label = stack.label
    plate_labels = stack.plate_labels

    time_limits = dict(SIGN_TIME_LIMITS_S)
    time_limits[1] = float(parking_time_limit_s)
    time_limits[2] = float(parking_time_limit_s)
    time_limits[3] = float(parking_time_limit_s)

    restriction = "no_stopping" if label in NO_STOPPING_LABELS else "no_parking"
    time_limit_s = 0.0 if label in NO_STOPPING_LABELS else time_limits.get(stack.main.class_id, parking_time_limit_s)
    applies_now = _date_rule_applies(label, now)

    # Mapping for signs 3.27-3.30:
    # - default: start_mode="from_sign", direction="forward";
    # - 8.2.2 / distance plate: direction="forward";
    # - 8.2.3: start_mode="to_sign", direction="backward";
    # - 8.2.4: start_mode="inside_zone", direction="both".
    start_mode = "from_sign"
    direction = "forward"

    metadata: dict[str, object] = {
        "rule_mapping": "ru_pdd_3_27_3_30_v1",
    }

    zone_distance_plates = [p for p in plate_labels if p in ZONE_DISTANCE_PLATE_LABELS]
    zone_end_plates = [p for p in plate_labels if p in ZONE_END_PLATE_LABELS]
    zone_inside_plates = [p for p in plate_labels if p in ZONE_INSIDE_PLATE_LABELS]
    zone_direction_plates = [p for p in plate_labels if p in ZONE_DIRECTION_PLATE_LABELS]

    if zone_distance_plates:
        metadata["zone_distance_plates"] = zone_distance_plates
    if zone_end_plates:
        metadata["zone_end_plates"] = zone_end_plates
    if zone_inside_plates:
        metadata["zone_inside_plates"] = zone_inside_plates
    if zone_direction_plates:
        metadata["zone_direction_plates"] = zone_direction_plates

    if zone_inside_plates:
        start_mode = "inside_zone"
        direction = "both"
    elif zone_end_plates:
        start_mode = "to_sign"
        direction = "backward"
    elif zone_distance_plates:
        start_mode = "from_sign"
        direction = "forward"

    exceptions = [item for item in plate_labels if item in EXCEPTION_PLATE_LABELS]
    if exceptions:
        metadata["exceptions"] = exceptions

    distance_m = _distance_from_metadata(stack.plates)
    if distance_m is not None:
        metadata["limited_by_plate"] = True
        metadata["distance_m"] = float(distance_m)

    return SignRule(
        sign_id=stack.main.class_id,
        sign_label=label,
        restriction=restriction,
        time_limit_s=float(time_limit_s),
        applies_now=applies_now,
        start_mode=start_mode,
        direction=direction,
        distance_m=distance_m,
        plate_labels=plate_labels,
        exception_labels=exceptions,
        metadata=metadata,
    )


def _date_rule_applies(label: str, now: datetime) -> bool:
    if label in ODD_DATE_SIGN_LABELS:
        return now.day % 2 == 1
    if label in EVEN_DATE_SIGN_LABELS:
        return now.day % 2 == 0
    return True


def _distance_from_metadata(plates: list[Detection]) -> float | None:
    for plate in plates:
        for key in ("distance_m", "zone_length_m", "length_m"):
            value = plate.metadata.get(key)
            if value is None:
                continue
            try:
                distance = float(value)
            except (TypeError, ValueError):
                continue
            if distance > 0.0:
                return distance

        # Conservative OCR fallback. Do not parse labels such as "8.2.1" as meters.
        for key in ("ocr_text", "text"):
            raw = plate.metadata.get(key)
            if not raw:
                continue
            text = str(raw).lower().replace(",", ".")
            match = re.search(r"(\d+(?:\.\d+)?)\s*(?:м|m|метр)", text)
            if match:
                try:
                    distance = float(match.group(1))
                except ValueError:
                    continue
                if distance > 0.0:
                    return distance

    return None