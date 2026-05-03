import argparse
from pathlib import Path

from parking_vision.common.video import resolve_source

from .constants import MODEL_PATH
from .reader import PlateReader
from .runner import VideoOutputConfig, run_plate_reading


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Russian license plate detection and reading module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Video file path or camera index (0, 1, ...)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to YOLO plate detection model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold (0.0-1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (0.0-1.0)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size (pixels)")
    parser.add_argument("--output", "-o", default="outputs/detected_plates.mp4", help="Annotated output video path")
    parser.add_argument("--csv", default="outputs/detected_plates.csv", help="CSV detections export path")
    parser.add_argument("--jsonl", default="outputs/detected_plates.jsonl", help="JSONL frame-by-frame export path")
    parser.add_argument("--show", "-s", action="store_true", help="Display processed frames in real-time")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader = PlateReader(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    output_config = VideoOutputConfig(
        annotated_video_path=Path(args.output),
        csv_path=Path(args.csv),
        jsonl_path=Path(args.jsonl),
    )
    Path("outputs").mkdir(exist_ok=True)
    run_plate_reading(
        source=resolve_source(args.source),
        reader=reader,
        output_config=output_config,
        show=args.show,
    )
    print(f"Annotated video saved to: {output_config.annotated_video_path}")
    print(f"CSV export saved to: {output_config.csv_path}")
    print(f"JSONL export saved to: {output_config.jsonl_path}")
