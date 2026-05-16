import atexit
import multiprocessing as mp
import os
import threading
from typing import Any, Tuple

import numpy as np


class PaddleOCRReader:
    _shared_client: Any | None = None
    _shared_process: mp.Process | None = None
    _shared_device: str | None = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self.connection, self.process, self.device = self._get_or_create_worker()

    @classmethod
    def _spawn_worker(cls) -> tuple[Any, mp.Process, str]:
        from .ocr_worker import run_ocr_worker

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        requested_device = os.environ.get("PARKING_VISION_PADDLEOCR_DEVICE", "gpu:0").strip() or "gpu:0"
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
            raise TimeoutError("Timed out while starting PaddleOCR worker process.")

        message = parent_conn.recv()
        status = message.get("status")
        if status == "ready":
            device = str(message.get("device", requested_device))
            print(f"[OCR] PaddleOCR worker ready on {device}")
            return parent_conn, process, device

        error = message.get("error", "Unknown OCR worker startup error.")
        process.join(timeout=5)
        raise RuntimeError(error)

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
        if self.process is None or not self.process.is_alive():
            self.connection, self.process, self.device = self._get_or_create_worker()

        self.connection.send({"type": "read", "image": image})
        response = self.connection.recv()
        if "error" in response:
            raise RuntimeError(response["error"])
        return str(response.get("text", "")), float(response.get("score", 0.0))


EasyOCRReader = PaddleOCRReader
