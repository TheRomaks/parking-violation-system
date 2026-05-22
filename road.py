import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


MODEL_NAME = "facebook/mask2former-swin-large-mapillary-vistas-semantic"


class RoadSegmenter:
    """
    Semantic road segmenter на Mask2Former Mapillary Vistas.

    Вход:
        frame: OpenCV BGR image, np.ndarray

    Выход:
        get_road_mask(frame) -> uint8 mask:
            0   = не дорога
            255 = road

    Дополнительно:
        get_sidewalk_mask(frame)
        get_curb_mask(frame)
        get_masks(frame)
    """

    def __init__(
        self,
        update_every_n_frames: int = 1,
        max_side: int = 768,
        device: str | None = None,
        use_half: bool = True,
        postprocess: bool = True,
        scene_cut_threshold: float = 34.0,
    ) -> None:
        self.update_every_n_frames = max(1, int(update_every_n_frames))
        self.max_side = int(max_side) if max_side else 0
        self.postprocess = bool(postprocess)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = bool(use_half and self.device == "cuda")

        self.frame_count = 0
        self.last_semantic_map: np.ndarray | None = None
        self.last_shape: tuple[int, int] | None = None
        self.last_scene_cut = False
        self.scene_cut_threshold = float(scene_cut_threshold)
        self._last_frame_signature: np.ndarray | None = None

        print(f"Loading model: {MODEL_NAME}")
        print(f"Device: {self.device}, half: {self.use_half}, max_side: {self.max_side}")

        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

        if self.use_half:
            self.model.half()

        self.id2label = {
            int(k): str(v)
            for k, v in self.model.config.id2label.items()
        }

        self.label2ids = self._build_label_index()

        self.road_ids = self._find_label_ids(["road"])
        self.sidewalk_ids = self._find_label_ids(["sidewalk"])
        self.curb_ids = self._find_label_ids(["curb"])

        print("Found classes:")
        self._print_found("road", self.road_ids)
        self._print_found("sidewalk", self.sidewalk_ids)
        self._print_found("curb", self.curb_ids)

        if not self.road_ids:
            raise RuntimeError(
                "Не найден класс road в id2label модели. "
                "Запусти с --print-labels и проверь названия классов."
            )

    @staticmethod
    def _norm_label(label: str) -> str:
        return label.lower().replace("_", " ").replace("-", " ").strip()

    def _build_label_index(self) -> dict[str, list[int]]:
        index: dict[str, list[int]] = {}

        for label_id, label_name in self.id2label.items():
            norm = self._norm_label(label_name)
            index.setdefault(norm, []).append(label_id)

        return index

    def _find_label_ids(self, names: list[str]) -> list[int]:
        result: list[int] = []

        wanted = {self._norm_label(name) for name in names}

        for norm_name, ids in self.label2ids.items():
            if norm_name in wanted:
                result.extend(ids)

        return sorted(set(result))

    def _print_found(self, name: str, ids: list[int]) -> None:
        if not ids:
            print(f"  {name}: NOT FOUND")
            return

        readable = ", ".join(f"{i}:{self.id2label[i]}" for i in ids)
        print(f"  {name}: {readable}")

    def print_labels(self) -> None:
        for label_id, label_name in sorted(self.id2label.items()):
            print(f"{label_id}: {label_name}")

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

    def _resize_for_inference(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame_bgr.shape[:2]

        if not self.max_side:
            return frame_bgr, 1.0

        longest = max(h, w)

        if longest <= self.max_side:
            return frame_bgr, 1.0

        scale = self.max_side / float(longest)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    @torch.inference_mode()
    def _predict_semantic_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        orig_h, orig_w = frame_bgr.shape[:2]

        frame_small_bgr, scale = self._resize_for_inference(frame_bgr)
        infer_h, infer_w = frame_small_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_small_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.use_half:
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.half()

        outputs = self.model(**inputs)

        semantic_map = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(infer_h, infer_w)],
        )[0]

        semantic_map = semantic_map.detach().cpu().numpy().astype(np.int32)

        if scale != 1.0:
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

    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--device", default=None, help="cuda / cpu")
    parser.add_argument("--no-half", action="store_true")
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
        update_every_n_frames=args.update_every,
        max_side=args.max_side,
        device=args.device,
        use_half=not args.no_half,
        postprocess=not args.no_postprocess,
    )

    if args.print_labels:
        segmenter.print_labels()
        return

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    is_video = input_path.suffix.lower() in video_exts

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
