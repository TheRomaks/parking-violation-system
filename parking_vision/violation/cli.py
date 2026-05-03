import argparse
from pathlib import Path

from parking_vision.common.video import resolve_source

from .pipeline import ViolationPipeline
from .runner import run_pipeline
from .types import VideoOutputConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified parking violation pipeline.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="outputs/violations_annotated.mp4")
    parser.add_argument("--violations-csv", dest="violations_csv_path", default="outputs/violations.csv")
    parser.add_argument("--jsonl", dest="jsonl_path", default="outputs/violation_pipeline.jsonl")
    parser.add_argument("--car-model", default="models/cars.pt")
    parser.add_argument("--sign-model", default="models/signs.pt")
    parser.add_argument("--plate-model", default="models/plates.pt")
    parser.add_argument("--stop-distance-threshold", type=float, default=12.0)
    parser.add_argument("--parking-time-limit", type=float, default=300.0)
    parser.add_argument("--stop-frames-threshold", type=int, default=15)
    parser.add_argument("--sign-interval-frames", type=int, default=1)
    parser.add_argument("--plate-interval-frames", type=int, default=4)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = ViolationPipeline(
        car_model_path=args.car_model,
        sign_model_path=args.sign_model,
        plate_model_path=args.plate_model,
        stop_distance_threshold_px=args.stop_distance_threshold,
        stop_frames_threshold=args.stop_frames_threshold,
        parking_time_limit_s=args.parking_time_limit,
        sign_interval_frames=args.sign_interval_frames,
        plate_interval_frames=args.plate_interval_frames,
    )
    output = VideoOutputConfig(
        annotated_video_path=Path(args.output),
        violations_csv_path=Path(args.violations_csv_path),
        jsonl_path=Path(args.jsonl_path),
    )
    run_pipeline(resolve_source(args.source), pipeline, output, args.show)
    print(f"Saved to: {output.annotated_video_path}, {output.violations_csv_path}, {output.jsonl_path}")
