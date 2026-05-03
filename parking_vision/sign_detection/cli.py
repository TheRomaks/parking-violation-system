import argparse
from pathlib import Path

from parking_vision.common.video import resolve_source

from .detector import SignDetector
from .runner import VideoOutputConfig, run_detection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Road sign detection module for the parking violation system."
    )
    parser.add_argument("--source", required=True, help="Video path or camera index.")
    parser.add_argument("--model", default="models/best.pt", help="Sign model weights.")
    parser.add_argument(
        "--output",
        default="outputs/detected_signs.mp4",
        help="Path to annotated output video.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default="outputs/detected_signs.csv",
        help="Flat CSV export for downstream modules.",
    )
    parser.add_argument(
        "--jsonl",
        dest="jsonl_path",
        default="outputs/detected_signs.jsonl",
        help="Frame-by-frame JSONL export for downstream modules.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display frames during processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = SignDetector(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    output_config = VideoOutputConfig(
        annotated_video_path=Path(args.output),
        csv_path=Path(args.csv_path),
        jsonl_path=Path(args.jsonl_path),
    )
    run_detection(
        source=resolve_source(args.source),
        detector=detector,
        output_config=output_config,
        show=args.show,
    )
    print(f"Annotated video saved to: {output_config.annotated_video_path}")
    print(f"CSV export saved to: {output_config.csv_path}")
    print(f"JSONL export saved to: {output_config.jsonl_path}")
