import cv2
import numpy as np


class RoadSegmenter:
    """
    Быстрая и достаточно стабильная сегментация дороги.
    Используется:
    - кэш маски на несколько кадров;
    - фильтрация по низкой насыщенности и средней яркости;
    - морфология;
    - сохранение крупных дорожных компонент, доходящих до нижней части кадра.
    """

    def __init__(self, update_every_n_frames: int = 3, downscale: float = 0.5) -> None:
        self.update_every_n_frames = max(1, int(update_every_n_frames))
        self.downscale = float(downscale)
        self.frame_count = 0
        self.last_mask: np.ndarray | None = None

    def get_road_mask(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1
        if self.last_mask is not None and self.frame_count % self.update_every_n_frames != 0:
            return self.last_mask

        h, w = frame.shape[:2]

        if self.downscale != 1.0:
            small = cv2.resize(
                frame,
                None,
                fx=self.downscale,
                fy=self.downscale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame

        hs, ws = small.shape[:2]

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        hsv_mask = cv2.inRange(hsv, (0, 0, 35), (180, 70, 235))
        gray_mask = cv2.inRange(gray, 30, 220)
        mask = cv2.bitwise_and(hsv_mask, gray_mask)

        mask[: int(hs * 0.28), :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            filtered = np.zeros_like(mask)
            min_area = max(500, int(hs * ws * 0.005))
            bottom_gate = int(hs * 0.70)

            for i in range(1, num_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                top = int(stats[i, cv2.CC_STAT_TOP])
                height = int(stats[i, cv2.CC_STAT_HEIGHT])
                bottom = top + height

                if area >= min_area and top > int(hs * 0.15) and bottom >= bottom_gate:
                    filtered[labels == i] = 255

            if np.any(filtered):
                mask = filtered

        mask = cv2.dilate(mask, kernel, iterations=1)

        if self.downscale != 1.0:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        self.last_mask = mask
        return mask

    @staticmethod
    def estimate_road_direction(road_mask: np.ndarray) -> np.ndarray | None:
        """
        Возвращает главный вектор направления дороги в координатах изображения:
        [dx, dy]. Если данных мало, возвращает None.
        """
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
        """
        Приближенный центр дороги по строке кадра.
        """
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
