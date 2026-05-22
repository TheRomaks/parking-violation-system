import os
import importlib
from typing import Any

import cv2
import numpy as np

from .constants import ALLOWLIST, RUS_PLATE_PATTERN
from .preprocessing import normalize_plate


def _extract_candidates(result: Any) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []

    if result is None:
        return candidates

    if isinstance(result, list):
        for item in result:
            candidates.extend(_extract_candidates(item))
        return candidates

    if isinstance(result, tuple):
        if len(result) >= 2 and isinstance(result[0], str):
            try:
                score = float(result[1])
            except (TypeError, ValueError):
                score = 0.0
            return [(result[0], score)]

        if len(result) >= 2 and isinstance(result[1], tuple):
            text_info = result[1]
            if len(text_info) >= 2 and isinstance(text_info[0], str):
                try:
                    score = float(text_info[1])
                except (TypeError, ValueError):
                    score = 0.0
                return [(text_info[0], score)]

    if isinstance(result, dict):
        text = result.get("rec_text") or result.get("text")
        score = result.get("rec_score") or result.get("score") or 0.0
        if isinstance(text, list):
            if text and isinstance(text[0], str):
                text = text[0]
            elif text:
                return _extract_candidates(text)
        if isinstance(score, list):
            score = score[0] if score else 0.0
        if isinstance(text, str):
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                score_value = 0.0
            return [(text, score_value)]

    return candidates


def _normalize_candidates(candidates: list[tuple[str, float]]) -> tuple[str, float]:
    if not candidates:
        return "", 0.0

    best_text = ""
    best_score = 0.0
    for text, score in candidates:
        normalized_text = normalize_plate(text)
        filtered_text = "".join(char for char in normalized_text if char in ALLOWLIST)
        if RUS_PLATE_PATTERN.match(filtered_text):
            return filtered_text, score
        if score > best_score:
            best_text = filtered_text
            best_score = score

    return best_text, best_score


def _build_reader(device: str):
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    _ensure_paddle_ready()
    from paddleocr import TextRecognition

    print(f"[OCR worker] TextRecognition device: {device}")
    return TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        device=device,
    )


def _ensure_paddle_ready() -> None:
    """Import Paddle fully before PaddleOCR touches it.

    PaddleOCR imports several Paddle submodules during its own initialization,
    so the worker checks Paddle before handing control to PaddleOCR. Do not
    delete Paddle submodules here: Paddle loads native .pyd modules such as
    paddle.base.libpaddle, and they are not safely reloadable inside one process.
    """

    try:
        paddle = importlib.import_module("paddle")
        importlib.import_module("paddle.tensor")
        importlib.import_module("paddle.base.libpaddle")
    except Exception as exc:
        raise RuntimeError(f"Paddle failed to initialize cleanly: {exc}") from exc

    if not hasattr(paddle, "tensor"):
        raise RuntimeError("Paddle failed to initialize cleanly: paddle.tensor is unavailable")


def _prepare_image_for_ocr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def run_ocr_worker(connection, requested_device: str) -> None:
    reader = None

    try:
        reader = _build_reader(requested_device)
        connection.send({"status": "ready", "device": requested_device})

        while True:
            message = connection.recv()
            message_type = message.get("type")

            if message_type == "close":
                break

            if message_type != "read":
                connection.send({"error": f"Unknown OCR worker message type: {message_type}"})
                continue

            image = message["image"]
            try:
                prepared_image = _prepare_image_for_ocr(image)
                results = reader.predict(prepared_image)
                text, score = _normalize_candidates(_extract_candidates(results))
                connection.send({"text": text, "score": score})
            except Exception as exc:
                connection.send({"error": str(exc)})
    except Exception as exc:
        try:
            connection.send({"status": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        try:
            if reader is not None and hasattr(reader, "close"):
                reader.close()
        except Exception:
            pass
        try:
            connection.close()
        except Exception:
            pass
