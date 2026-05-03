import re

import cv2
import numpy as np

from .constants import CONF_THRESHOLD, MAX_AR, MIN_AREA, MIN_AR


def normalize_plate(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"[^A-Za-zА-Я0-9]", "", text.upper())
    replacements = {
        "А": "A",
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "У": "Y",
        "Х": "X",
    }
    return "".join(replacements.get(char, char) for char in text)


def preprocess(crop: np.ndarray) -> np.ndarray | None:
    if crop is None or crop.size == 0:
        return None

    pad = 10
    crop = cv2.copyMakeBorder(
        crop,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)
    return gray


def crop_bbox(frame: np.ndarray, box: list[float]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def is_valid_plate(box: list[float], conf: float) -> bool:
    if conf < CONF_THRESHOLD:
        return False

    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return False

    area = width * height
    aspect_ratio = width / height
    if area < MIN_AREA:
        return False
    if aspect_ratio < MIN_AR or aspect_ratio > MAX_AR:
        return False

    return True
