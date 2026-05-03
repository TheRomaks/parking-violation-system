import argparse
from pathlib import Path

from parking_vision.common.video import resolve_source

from .detector import CarTracker
from .runner import VideoOutputConfig, run_tracking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Car detection and tracking module for the parking violation system."
    )
    parser.add_argument("--source", required=True, help="Video path or camera index.")
    parser.add_argument("--model", default="models/cars.pt", help="Car model weights.")
    parser.add_argument(
        "--output",
        default="outputs/tracked_cars.mp4",
        help="Path to annotated output video.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default="outputs/tracked_cars.csv",
        help="Flat CSV export for downstream modules.",
    )
    parser.add_argument(
        "--jsonl",
        dest="jsonl_path",
        default="outputs/tracked_cars.jsonl",
        help="Frame-by-frame JSONL export for downstream modules.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display frames during processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracker = CarTracker(
        model_path=args.model,
        tracker_config=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    output_config = VideoOutputConfig(
        annotated_video_path=Path(args.output),
        csv_path=Path(args.csv_path),
        jsonl_path=Path(args.jsonl_path),
    )
    run_tracking(
        source=resolve_source(args.source),
        tracker=tracker,
        output_config=output_config,
        show=args.show,
    )
    print(f"Annotated video saved to: {output_config.annotated_video_path}")
    print(f"CSV export saved to: {output_config.csv_path}")
    print(f"JSONL export saved to: {output_config.jsonl_path}")
