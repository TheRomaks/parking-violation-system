import atexit
import multiprocessing as mp
import os
import threading
from typing import Any, Tuple

import numpy as np

from .constants import ALLOWLIST, RUS_PLATE_PATTERN
from .preprocessing import normalize_plate


class PaddleOCRReader:
    _shared_client: Any | None = None
    _shared_process: mp.Process | None = None
    _shared_device: str | None = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        backend = os.environ.get("PARKING_VISION_OCR_BACKEND", "auto").strip().lower() or "auto"
        self.fallback_reader: EasyOCRReader | None = None

        if backend == "easyocr":
            self.connection = None
            self.process = None
            self.device = "easyocr"
            self.fallback_reader = EasyOCRReader()
            return

        if backend not in {"auto", "paddle", "paddleocr"}:
            raise ValueError(
                "Unsupported PARKING_VISION_OCR_BACKEND value. "
                "Use 'auto', 'paddle', or 'easyocr'."
            )

        try:
            self.connection, self.process, self.device = self._get_or_create_worker()
        except RuntimeError as exc:
            if backend in {"paddle", "paddleocr"}:
                raise

            print(f"[OCR] PaddleOCR unavailable, falling back to EasyOCR: {exc}")
            self.connection = None
            self.process = None
            self.device = "easyocr"
            self.fallback_reader = EasyOCRReader()

    @classmethod
    def _spawn_worker_for_device(cls, requested_device: str) -> tuple[Any, mp.Process, str]:
        from .ocr_worker import run_ocr_worker

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(
            target=run_ocr_worker,
            args=(child_conn, requested_device),
            daemon=True,
        )
        process.start()
        child_conn.close()

        if not parent_conn.poll(180):
            process.kill()
            process.join(timeout=5)
            parent_conn.close()
            raise TimeoutError("Timed out while starting PaddleOCR worker process.")

        message = parent_conn.recv()
        status = message.get("status")
        if status == "ready":
            device = str(message.get("device", requested_device))
            print(f"[OCR] PaddleOCR worker ready on {device}")
            return parent_conn, process, device

        error = message.get("error", "Unknown OCR worker startup error.")
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        parent_conn.close()
        raise RuntimeError(error)

    @classmethod
    def _spawn_worker(cls) -> tuple[Any, mp.Process, str]:
        requested_device = os.environ.get("PARKING_VISION_PADDLEOCR_DEVICE", "gpu:0").strip() or "gpu:0"

        try:
            return cls._spawn_worker_for_device(requested_device)
        except Exception as gpu_exc:
            if not requested_device.lower().startswith("gpu"):
                raise

            print(f"[OCR] PaddleOCR worker failed on {requested_device}, retrying on CPU: {gpu_exc}")
            try:
                return cls._spawn_worker_for_device("cpu")
            except Exception as cpu_exc:
                raise RuntimeError(
                    "PaddleOCR failed to start on GPU and CPU. "
                    f"GPU error: {gpu_exc}. CPU error: {cpu_exc}"
                ) from cpu_exc

    @classmethod
    def _get_or_create_worker(cls) -> tuple[Any, mp.Process, str]:
        if (
            cls._shared_client is not None
            and cls._shared_process is not None
            and cls._shared_process.is_alive()
            and cls._shared_device is not None
        ):
            return cls._shared_client, cls._shared_process, cls._shared_device

        with cls._init_lock:
            if (
                cls._shared_client is not None
                and cls._shared_process is not None
                and cls._shared_process.is_alive()
                and cls._shared_device is not None
            ):
                return cls._shared_client, cls._shared_process, cls._shared_device

            cls._shared_client, cls._shared_process, cls._shared_device = cls._spawn_worker()
            atexit.register(cls.close_shared_worker)
            return cls._shared_client, cls._shared_process, cls._shared_device

    @classmethod
    def close_shared_worker(cls) -> None:
        connection = cls._shared_client
        process = cls._shared_process

        cls._shared_client = None
        cls._shared_process = None
        cls._shared_device = None

        if connection is not None:
            try:
                connection.send({"type": "close"})
            except Exception:
                pass
            try:
                connection.close()
            except Exception:
                pass

        if process is not None and process.is_alive():
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)

    def read(self, image: np.ndarray) -> Tuple[str, float]:
        if self.fallback_reader is not None:
            return self.fallback_reader.read(image)

        if self.process is None or not self.process.is_alive():
            self.connection, self.process, self.device = self._get_or_create_worker()

        self.connection.send({"type": "read", "image": image})
        response = self.connection.recv()
        if "error" in response:
            raise RuntimeError(response["error"])
        return str(response.get("text", "")), float(response.get("score", 0.0))


class EasyOCRReader:
    _shared_reader: Any | None = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self.reader = self._get_or_create_reader()

    @classmethod
    def _get_or_create_reader(cls) -> Any:
        if cls._shared_reader is not None:
            return cls._shared_reader

        with cls._init_lock:
            if cls._shared_reader is not None:
                return cls._shared_reader

            import torch
            import easyocr

            use_gpu = torch.cuda.is_available()
            print(f"[OCR] EasyOCR reader ready on {'gpu' if use_gpu else 'cpu'}")
            cls._shared_reader = easyocr.Reader(["en"], gpu=use_gpu)
            return cls._shared_reader

    def read(self, image: np.ndarray) -> Tuple[str, float]:
        results = self.reader.readtext(
            image,
            detail=1,
            allowlist=ALLOWLIST,
            paragraph=False,
        )

        best_text = ""
        best_score = 0.0
        for result in results:
            if len(result) < 3:
                continue

            text = normalize_plate(str(result[1]))
            filtered_text = "".join(char for char in text if char in ALLOWLIST)
            try:
                score = float(result[2])
            except (TypeError, ValueError):
                score = 0.0

            if RUS_PLATE_PATTERN.match(filtered_text):
                return filtered_text, score
            if score > best_score:
                best_text = filtered_text
                best_score = score

        return best_text, best_score
