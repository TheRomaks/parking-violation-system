from typing import Tuple

import easyocr
import numpy as np
import torch

from .constants import ALLOWLIST, RUS_PLATE_PATTERN
from .preprocessing import normalize_plate


class EasyOCRReader:
    def __init__(self) -> None:
        gpu = torch.cuda.is_available()
        print(f"[OCR] GPU: {gpu}")
        self.reader = easyocr.Reader(["ru", "en"], gpu=gpu, verbose=False)

    def read(self, image: np.ndarray) -> Tuple[str, float]:
        results = self.reader.readtext(
            image,
            detail=1,
            allowlist=ALLOWLIST,
            paragraph=False,
        )
        if not results:
            return "", 0.0

        best_text = ""
        best_score = 0.0
        for _, text, score in results:
            normalized_text = normalize_plate(text)
            if RUS_PLATE_PATTERN.match(normalized_text):
                return normalized_text, score
            if score > best_score:
                best_text = normalized_text
                best_score = score

        return best_text, best_score
