from parking_vision.common.video import open_writer, resolve_source
from parking_vision.sign_detection import (
    MODULE_NAME,
    SignDetector,
    VideoOutputConfig,
    main,
    parse_args,
    run_detection,
)

__all__ = [
    "MODULE_NAME",
    "SignDetector",
    "VideoOutputConfig",
    "main",
    "open_writer",
    "parse_args",
    "resolve_source",
    "run_detection",
]


if __name__ == "__main__":
    main()
