import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_MODEL_PATH = "models/segmentation.pt"


class RoadSegmenter:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        update_every_n_frames: int = 1,
        max_side: int = 640,
        device: str | None = None,
        use_half: bool = False,
        postprocess: bool = True,
        scene_cut_threshold: float = 34.0,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")

        self.update_every_n_frames = max(1, int(update_every_n_frames))
        self.max_side = int(max_side) if max_side else 640
        self.device = device
        self.use_half = bool(use_half)
        self.postprocess = bool(postprocess)

        self.frame_count = 0
        self.last_semantic_map: np.ndarray | None = None
        self.last_shape: tuple[int, int] | None = None
        self.last_scene_cut = False
        self.scene_cut_threshold = float(scene_cut_threshold)
        self._last_frame_signature: np.ndarray | None = None

        print(f"Loading model: {self.model_path}")
        print(f"Device: {self.device or 'auto'}, half: {self.use_half}, imgsz: {self.max_side}")

        self.model = YOLO(str(self.model_path), task="semantic")

        self.class_names = self._get_class_names()

        # Автоматически ищем классы по именам.
        # Если names не найдены, используем стандартную 4-классовую схему.
        self.background_ids = self._find_label_ids(["background"]) or [0]
        self.curb_ids = self._find_label_ids(["curb"]) or [1]
        self.road_ids = self._find_label_ids(["road"]) or [2]
        self.sidewalk_ids = self._find_label_ids(["sidewalk"]) or [3]

        print("Found classes:")
        self._print_found("background", self.background_ids)
        self._print_found("curb", self.curb_ids)
        self._print_found("road", self.road_ids)
        self._print_found("sidewalk", self.sidewalk_ids)

        if not self.road_ids:
            raise RuntimeError("Не найден класс road в модели.")

    def _get_class_names(self) -> dict[int, str]:
        names = getattr(self.model, "names", None)

        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}

        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}

        return {
            0: "background",
            1: "curb",
            2: "road",
            3: "sidewalk",
        }

    @staticmethod
    def _norm_label(label: str) -> str:
        return label.lower().replace("_", " ").replace("-", " ").strip()

    def _find_label_ids(self, names: list[str]) -> list[int]:
        wanted = {self._norm_label(name) for name in names}
        result: list[int] = []

        for class_id, class_name in self.class_names.items():
            if self._norm_label(class_name) in wanted:
                result.append(class_id)

        return sorted(set(result))

    def _print_found(self, name: str, ids: list[int]) -> None:
        if not ids:
            print(f"  {name}: NOT FOUND")
            return

        readable = ", ".join(
            f"{class_id}:{self.class_names.get(class_id, 'unknown')}"
            for class_id in ids
        )
        print(f"  {name}: {readable}")

    def print_labels(self) -> None:
        for class_id, class_name in sorted(self.class_names.items()):
            print(f"{class_id}: {class_name}")

    def reset(self) -> None:
        self.frame_count = 0
        self.last_semantic_map = None
        self.last_shape = None
        self.last_scene_cut = False
        self._last_frame_signature = None

    @staticmethod
    def _frame_signature(frame_bgr: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame_bgr, (32, 18), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float32)

    def _update_scene_cut_state(self, frame_bgr: np.ndarray) -> None:
        signature = self._frame_signature(frame_bgr)
        self.last_scene_cut = False

        if self._last_frame_signature is not None:
            diff = float(np.mean(np.abs(signature - self._last_frame_signature)))

            if diff >= self.scene_cut_threshold:
                self.last_scene_cut = True
                self.last_semantic_map = None
                self.last_shape = None

        self._last_frame_signature = signature

    @staticmethod
    def _extract_semantic_map(result: Any) -> np.ndarray:
        """
        Достаёт semantic map из результата Ultralytics.

        Ожидается result.semantic_mask.
        """
        semantic_mask = getattr(result, "semantic_mask", None)

        if semantic_mask is None:
            raise RuntimeError(
                "Модель не вернула semantic_mask. "
                "Проверь, что segmentation.pt обучена как YOLO semantic segmentation "
                "и загружается с task='semantic'."
            )

        data = semantic_mask.data if hasattr(semantic_mask, "data") else semantic_mask

        if hasattr(data, "detach"):
            data = data.detach()

        if hasattr(data, "cpu"):
            data = data.cpu()

        if hasattr(data, "numpy"):
            data = data.numpy()

        semantic_map = np.asarray(data)

        if semantic_map.ndim == 3:
            semantic_map = np.squeeze(semantic_map)

        return semantic_map.astype(np.int32)

    def _predict_semantic_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        orig_h, orig_w = frame_bgr.shape[:2]

        predict_kwargs = {
            "source": frame_bgr,
            "imgsz": self.max_side,
            "verbose": False,
        }

        if self.device is not None:
            predict_kwargs["device"] = self.device

        if self.use_half:
            predict_kwargs["half"] = True

        results = self.model.predict(**predict_kwargs)
        result = results[0]

        semantic_map = self._extract_semantic_map(result)

        if semantic_map.shape[:2] != (orig_h, orig_w):
            semantic_map = cv2.resize(
                semantic_map,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

        return semantic_map

    def get_semantic_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Возвращает карту классов HxW, где значение пикселя = class_id модели.
        """
        self._update_scene_cut_state(frame_bgr)
        self.frame_count += 1

        shape = frame_bgr.shape[:2]

        can_reuse = (
            self.last_semantic_map is not None
            and self.last_shape == shape
            and self.frame_count % self.update_every_n_frames != 0
        )

        if can_reuse:
            return self.last_semantic_map

        semantic_map = self._predict_semantic_map(frame_bgr)

        self.last_semantic_map = semantic_map
        self.last_shape = shape

        return semantic_map

    def _mask_from_ids(self, semantic_map: np.ndarray, label_ids: list[int]) -> np.ndarray:
        if not label_ids:
            return np.zeros(semantic_map.shape, dtype=np.uint8)

        mask = np.isin(semantic_map, label_ids).astype(np.uint8) * 255

        if self.postprocess:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    def get_road_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        semantic_map = self.get_semantic_map(frame_bgr)
        return self._mask_from_ids(semantic_map, self.road_ids)

    def get_sidewalk_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        semantic_map = self.get_semantic_map(frame_bgr)
        return self._mask_from_ids(semantic_map, self.sidewalk_ids)

    def get_curb_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        semantic_map = self.get_semantic_map(frame_bgr)
        return self._mask_from_ids(semantic_map, self.curb_ids)

    def get_masks(self, frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
        semantic_map = self.get_semantic_map(frame_bgr)

        return {
            "road": self._mask_from_ids(semantic_map, self.road_ids),
            "sidewalk": self._mask_from_ids(semantic_map, self.sidewalk_ids),
            "curb": self._mask_from_ids(semantic_map, self.curb_ids),
        }

    @staticmethod
    def estimate_road_direction(road_mask: np.ndarray) -> np.ndarray | None:
        ys, xs = np.where(road_mask > 0)

        if len(xs) < 200:
            return None

        n = min(5000, len(xs))
        idx = np.linspace(0, len(xs) - 1, n).astype(int)
        pts = np.column_stack((xs[idx], ys[idx])).astype(np.float32)

        pts -= pts.mean(axis=0, keepdims=True)
        cov = np.cov(pts, rowvar=False)

        if cov.shape != (2, 2):
            return None

        vals, vecs = np.linalg.eigh(cov)
        direction = vecs[:, int(np.argmax(vals))]
        norm = float(np.linalg.norm(direction))

        if norm < 1e-6:
            return None

        return (direction / norm).astype(np.float32)

    @staticmethod
    def road_center_x_at_y(road_mask: np.ndarray, y: int, band: int = 10) -> float | None:
        h, _ = road_mask.shape[:2]

        y1 = max(0, y - band)
        y2 = min(h, y + band + 1)

        if y1 >= y2:
            return None

        band_mask = road_mask[y1:y2, :]
        cols = np.where(np.any(band_mask > 0, axis=0))[0]

        if len(cols) == 0:
            return None

        return float((int(cols[0]) + int(cols[-1])) / 2.0)


def overlay_masks(frame: np.ndarray, masks: dict[str, np.ndarray], alpha: float = 0.45) -> np.ndarray:
    """
    Визуализация:
      road     = green
      sidewalk = red
      curb     = yellow
    """
    overlay = frame.copy()

    road = masks.get("road")
    sidewalk = masks.get("sidewalk")
    curb = masks.get("curb")

    if road is not None:
        overlay[road > 0] = (0, 255, 0)

    if sidewalk is not None:
        overlay[sidewalk > 0] = (0, 0, 255)

    if curb is not None:
        overlay[curb > 0] = (0, 255, 255)

    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)


def process_image(
    input_path: str,
    output_path: str,
    segmenter: RoadSegmenter,
    show_all: bool = True,
) -> None:
    frame = cv2.imread(input_path)

    if frame is None:
        print(f"Ошибка: не удалось открыть изображение: {input_path}", file=sys.stderr)
        return

    if show_all:
        masks = segmenter.get_masks(frame)
        vis = overlay_masks(frame, masks)
    else:
        road_mask = segmenter.get_road_mask(frame)
        vis = frame.copy()
        vis[road_mask > 0] = (0, 255, 0)

    cv2.imwrite(output_path, vis)
    print(f"Saved: {output_path}")


def process_video(
    input_path: str,
    output_path: str,
    segmenter: RoadSegmenter,
    show_all: bool = True,
    side_by_side: bool = False,
    max_frames: int | None = None,
) -> None:
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео: {input_path}", file=sys.stderr)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if side_by_side:
        out_size = (width * 2, height)
    else:
        out_size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, out_size)

    frame_id = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_id += 1

        if max_frames is not None and frame_id > max_frames:
            break

        print(f"\rFrame {frame_id}", end="", flush=True)

        if show_all:
            masks = segmenter.get_masks(frame)
            vis = overlay_masks(frame, masks)
        else:
            road_mask = segmenter.get_road_mask(frame)
            vis = frame.copy()
            vis[road_mask > 0] = (0, 255, 0)

        if side_by_side:
            combined = np.hstack([frame, vis])
        else:
            combined = vis

        out.write(combined)

    cap.release()
    out.release()

    print(f"\nSaved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Путь к изображению или видео")
    parser.add_argument("-o", "--output", default=None, help="Путь для результата")

    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Путь к модели segmentation.pt")

    parser.add_argument("--max-side", type=int, default=640, help="imgsz для YOLO inference")
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--device", default=None, help="0 / 1 / cuda / cpu")
    parser.add_argument("--half", action="store_true", help="Использовать half precision")
    parser.add_argument("--no-postprocess", action="store_true")

    parser.add_argument("--road-only", action="store_true", help="Показывать только road")
    parser.add_argument("--side-by-side", action="store_true", help="Для видео: оригинал слева, маска справа")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--print-labels", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Файл не найден: {input_path}", file=sys.stderr)
        return

    segmenter = RoadSegmenter(
        model_path=args.model,
        update_every_n_frames=args.update_every,
        max_side=args.max_side,
        device=args.device,
        use_half=args.half,
        postprocess=not args.no_postprocess,
    )

    if args.print_labels:
        segmenter.print_labels()
        return

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    suffix = input_path.suffix.lower()

    is_video = suffix in video_exts
    is_image = suffix in image_exts

    if not is_video and not is_image:
        print(f"Неизвестный тип файла: {input_path}", file=sys.stderr)
        return

    if args.output is None:
        if is_video:
            output_path = str(input_path.with_name(input_path.stem + "_seg.mp4"))
        else:
            output_path = str(input_path.with_name(input_path.stem + "_seg.png"))
    else:
        output_path = args.output

    if is_video:
        process_video(
            input_path=str(input_path),
            output_path=output_path,
            segmenter=segmenter,
            show_all=not args.road_only,
            side_by_side=args.side_by_side,
            max_frames=args.max_frames,
        )
    else:
        process_image(
            input_path=str(input_path),
            output_path=output_path,
            segmenter=segmenter,
            show_all=not args.road_only,
        )


if __name__ == "__main__":
    main()