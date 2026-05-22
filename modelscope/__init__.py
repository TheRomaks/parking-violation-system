"""Minimal ModelScope stub for PaddleX official-model downloads.

This project does not use ModelScope as a model source, but recent
`paddleocr/paddlex` imports it unconditionally on startup. The real
ModelScope package pulls in PyTorch, which conflicts with Paddle's GPU
runtime on Windows in this app. A tiny local stub keeps PaddleOCR on the
official download path without importing PyTorch during startup.
"""


def snapshot_download(*args, **kwargs):
    raise RuntimeError(
        "ModelScope downloads are disabled in this project. "
        "Use Paddle official model sources instead."
    )
