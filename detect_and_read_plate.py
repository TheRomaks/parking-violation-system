import argparse
import re
from pathlib import Path
from typing import Any
from dataclasses import dataclass
import cv2
from ultralytics import YOLO

MODEL_PATH = "models/plates.pt"

CONF_THRESHOLD = 0.5
MIN_AREA = 800

MIN_AR = 2.0
MAX_AR = 6.0

# OCR
ALLOWLIST = "ABEKMHOPCTYX0123456789"

# РФ номер
RUS_PLATE_PATTERN = re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$")

@dataclass(slots=True)
class VideoOutputConfig:
    annotated_video_path: Path
    csv_path: Path
    jsonl_path: Path

def resolve_source(source_value: str) -> int | str:
    return int(source_value) if source_value.isdigit() else source_value

def open_writer(output_path: Path, capture: cv2.VideoCapture, frame):
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    return cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for plate detection module."""
    
    parser = argparse.ArgumentParser(
        description="Russian license plate detection and reading module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--source",
        required=True,
        help="Video file path or camera index (0, 1, ...)"
    )
    
    parser.add_argument(
        "--model",
        default="models/plates.pt",
        help="Path to YOLO plate detection model weights"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size (pixels)"
    )

    parser.add_argument(
        "--output", "-o",
        default="outputs/detected_plates.mp4",
        help="Annotated output video path"
    )
    parser.add_argument(
        "--csv",
        default="outputs/detected_plates.csv",
        help="CSV detections export path"
    )
    parser.add_argument(
        "--jsonl",
        default="outputs/detected_plates.jsonl",
        help="JSONL frame-by-frame export path"
    )
    
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display processed frames in real-time"
    )
    
    return parser.parse_args()


def run_plate_reading(
    source: int | str,
    reader,
    output_config: VideoOutputConfig,
    show: bool = False,
):

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read first frame")

    writer = open_writer(output_config.annotated_video_path, cap, first_frame)

    frame_index = 0

    while True:
        frame = first_frame if frame_index == 0 else None

        if frame is None:
            ok, frame = cap.read()
            if not ok:
                break

        plates = reader.process_frame(frame)

        for p in plates:
            x1, y1, x2, y2 = map(int, p["bbox"])

            color = (0, 255, 0) if p["valid"] else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{p['text']} ({p['ocr_conf']:.2f})"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        writer.write(frame)

        if show:
            cv2.imshow("Plate Reader", frame)
            if cv2.waitKey(1) == 27:
                break

        frame_index += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

class EasyOCRReader:
    def __init__(self):
        import easyocr
        import torch

        gpu = torch.cuda.is_available()
        print(f"[OCR] GPU: {gpu}")

        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read(self, image):
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
            norm = normalize_plate(text)

            if RUS_PLATE_PATTERN.match(norm):
                return norm, score

            if score > best_score:
                best_text = norm
                best_score = score

        return best_text, best_score



def normalize_plate(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"[^A-Za-zА-Я0-9]", "", text.upper())

    replacements = {
        "А": "A", "В": "B", "Е": "E", "К": "K",
        "М": "M", "Н": "H", "О": "O", "Р": "P",
        "С": "C", "Т": "T", "У": "Y", "Х": "X",
    }

    text = "".join(replacements.get(c, c) for c in text)

    return text


def preprocess(crop):
    if crop is None or crop.size == 0:
        return None

    pad = 10
    crop = cv2.copyMakeBorder(
        crop, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray, None,
        fx=2.5, fy=2.5,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.equalizeHist(gray)

    return gray


def crop_bbox(frame, box):
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def is_valid_plate(box, conf):
    if conf < CONF_THRESHOLD:
        return False

    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return False

    area = w * h
    ar = w / h

    if area < MIN_AREA:
        return False

    if ar < MIN_AR or ar > MAX_AR:
        return False

    return True

class PlateReader:

    def __init__(
        self,
        model_path=MODEL_PATH,
        conf=0.3,
        iou=0.5,
        imgsz=1280
    ):
        self.model = YOLO(model_path)
        self.ocr = EasyOCRReader()

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def process_frame(self, frame):

        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz
        )[0]

        plates = []

        if results.boxes is None:
            return plates

        for box, conf in zip(results.boxes.xyxy, results.boxes.conf):

            box = box.cpu().tolist()
            conf = float(conf)

            if not is_valid_plate(box, conf):
                continue

            crop = crop_bbox(frame, box)
            proc = preprocess(crop)

            if proc is None:
                continue

            text, score = self.ocr.read(proc)

            if score < 0.4:
                continue

            plates.append({
                "bbox": box,
                "conf": conf,
                "text": text,
                "ocr_conf": score,
                "valid": bool(RUS_PLATE_PATTERN.match(text))
            })

        return plates


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
    run_plate_reading(
        source=resolve_source(args.source),
        reader=reader,
        output_config=output_config,
        show=args.show,
    )
    print(f"Annotated video saved to: {output_config.annotated_video_path}")
    print(f"CSV export saved to: {output_config.csv}")
    print(f"JSONL export saved to: {output_config.jsonl}")
    print(f"OCR backend: {reader.ocr_backend.name}")

if __name__ == "__main__":
    main()