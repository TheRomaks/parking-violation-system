from pathlib import Path
from typing import Any

import cv2

from perception_types import BoundingBox, Detection, FrameDetections


MODULE_NAME = "car_tracker"


class CarTracker:
    def __init__(
        self,
        model_path: str = "models/cars.pt",
        tracker_config: str = "bytetrack.yaml",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1280,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.tracker_config = tracker_config
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def process_frame(
        self,
        frame: Any,
        frame_index: int,
        timestamp_ms: float | None = None,
    ) -> FrameDetections:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        detections: list[Detection] = []

        if boxes is not None and len(boxes) > 0:
            xyxy_list = boxes.xyxy.cpu().tolist()
            conf_list = boxes.conf.cpu().tolist()
            cls_list = boxes.cls.cpu().tolist()
            id_list = (
                boxes.id.int().cpu().tolist()
                if boxes.id is not None
                else [None] * len(xyxy_list)
            )

            for box, conf, cls_id, track_id in zip(
                xyxy_list,
                conf_list,
                cls_list,
                id_list,
            ):
                class_id = int(cls_id)
                class_name = result.names.get(class_id, str(class_id))
                bbox = BoundingBox(*box)
                detections.append(
                    Detection(
                        module=MODULE_NAME,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(conf),
                        bbox=bbox,
                        track_id=None if track_id is None else int(track_id),
                        metadata={
                            "model": self.model_path.name,
                            "tracker": self.tracker_config,
                        },
                    )
                )

        return FrameDetections(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            detections=detections,
        )

    @staticmethod
    def annotate_frame(frame: Any, frame_result: FrameDetections) -> Any:
        annotated = frame.copy()
        for detection in frame_result.detections:
            x1, y1, x2, y2 = detection.bbox.to_int_tuple()
            color = (0, 200, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            track_label = detection.track_id if detection.track_id is not None else "NA"
            label = (
                f"{detection.class_name} "
                f"ID:{track_label} "
                f"conf:{detection.confidence:.2f}"
            )
            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )
            text_y = max(y1 - 10, text_h + 6)
            cv2.rectangle(
                annotated,
                (x1, text_y - text_h - 6),
                (x1 + text_w + 8, text_y),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, text_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
        return annotated
