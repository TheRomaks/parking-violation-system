import argparse
import cv2
import numpy as np
from pathlib import Path
import sys


SCENE_CUT_BHATTACHARYYA = 0.46
SCENE_CUT_MEAN_DIFF = 0.13
HORIZON_RATIO = 0.34
BOTTOM_SAMPLE_RATIO = 0.22
MIN_ROAD_WIDTH_PX = 24
RIGHT_LANE_WIDTH_FRACTION = 0.48
RIGHT_LANE_MIN_WIDTH_PX = 30
RIGHT_LANE_MAX_WIDTH_RATIO = 0.62
RIGHT_EDGE_MARGIN_PX = 2
RIGHT_EDGE_INSET_FRAC = 0.08
RIGHT_EDGE_INSET_MIN_PX = 5
RIGHT_EDGE_INSET_MAX_PX = 22
ANCHOR_MIN_X_RATIO = 0.12
ANCHOR_MAX_X_RATIO = 0.84
RIGHT_BORDER_PENALTY_RATIO = 0.95
INNER_MARKING_MIN_RATIO = 0.34
INNER_MARKING_MAX_RATIO = 0.64


def _smooth_1d(arr: np.ndarray, k: int = 9) -> np.ndarray:
    if arr is None or len(arr) < 3:
        return arr
    k = max(3, int(k) | 1)
    pad = k // 2
    padded = np.pad(arr.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(padded, kernel, mode="valid")


def _find_runs(xs: np.ndarray) -> list[tuple[int, int]]:
    if xs.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(xs[0])
    prev = int(xs[0])
    for value in xs[1:]:
        value = int(value)
        if value != prev + 1:
            runs.append((start, prev))
            start = value
        prev = value
    runs.append((start, prev))
    return runs


class SceneCutDetector:
    def __init__(
        self,
        threshold: float = SCENE_CUT_BHATTACHARYYA,
        mean_diff: float = SCENE_CUT_MEAN_DIFF,
        min_gap: int = 3,
    ) -> None:
        self.threshold = float(threshold)
        self.mean_diff = float(mean_diff)
        self.min_gap = int(min_gap)
        self.prev_hist: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None
        self.last_cut_frame = -10_000

    def reset(self) -> None:
        self.prev_hist = None
        self.prev_gray = None
        self.last_cut_frame = -10_000

    def update(self, frame_bgr: np.ndarray, frame_id: int) -> bool:
        small = cv2.resize(frame_bgr, (96, 54), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)

        if self.prev_hist is None or self.prev_gray is None:
            self.prev_hist = hist
            self.prev_gray = gray
            return False

        hist_dist = float(cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        mean_diff = float(np.mean(cv2.absdiff(gray, self.prev_gray)) / 255.0)

        self.prev_hist = hist
        self.prev_gray = gray

        if frame_id - self.last_cut_frame < self.min_gap:
            return False

        if hist_dist >= self.threshold and mean_diff >= self.mean_diff:
            self.last_cut_frame = frame_id
            return True
        return False


class RoadSegmenter:
    """
    Segmenter that keeps only the drivable carriageway.

    The mask is built from adaptive appearance cues sampled at the bottom of
    the frame, then converted into a perspective-aware corridor by tracking a
    bottom-connected run through the image depth.
    """

    def __init__(
        self,
        update_every_n_frames: int = 1,
        downscale: float = 0.5,
        temporal_alpha: float = 0.30,
    ) -> None:
        self.update_every_n_frames = max(1, int(update_every_n_frames))
        self.downscale = float(downscale)
        self.temporal_alpha = float(np.clip(temporal_alpha, 0.0, 1.0))
        self.frame_count = 0
        self.last_mask: np.ndarray | None = None
        self.last_small_mask: np.ndarray | None = None
        self.last_scene_cut = False
        self._cut_detector = SceneCutDetector()

    def reset(self) -> None:
        self.last_mask = None
        self.last_small_mask = None
        self.last_scene_cut = False
        self._cut_detector.reset()

    def get_road_mask(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1
        if self.last_mask is not None and self.frame_count % self.update_every_n_frames != 0:
            return self.last_mask

        scene_cut = self._cut_detector.update(frame, self.frame_count)
        self.last_scene_cut = scene_cut
        if scene_cut:
            self.last_mask = None
            self.last_small_mask = None

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
            small = frame.copy()

        confidence = self._build_confidence_map(small)
        mask_small = self._extract_corridor_mask(confidence, scene_cut=scene_cut)

        if self.last_small_mask is not None and not scene_cut and self.last_small_mask.shape == mask_small.shape:
            prev = self.last_small_mask.astype(np.float32)
            curr = mask_small.astype(np.float32)
            blended = (1.0 - self.temporal_alpha) * curr + self.temporal_alpha * prev
            mask_small = (blended >= 0.5).astype(np.uint8)
            mask_small = self._finalize_mask(mask_small)

        if self.downscale != 1.0:
            mask = cv2.resize(mask_small * 255, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8) * 255
        else:
            mask = mask_small.astype(np.uint8) * 255

        self.last_small_mask = mask_small.copy()
        self.last_mask = mask
        return mask

    def _build_confidence_map(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        grad_x = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        lap = np.abs(cv2.Laplacian(gray_blur, cv2.CV_32F, ksize=3))
        texture = cv2.GaussianBlur(grad + 0.6 * lap, (5, 5), 0)

        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)
        lab_f = lab.astype(np.float32)

        y0 = int(h * (1.0 - BOTTOM_SAMPLE_RATIO))
        x0 = int(w * 0.18)
        x1 = int(w * 0.78)
        sample_sat = sat[y0:, x0:x1]
        sample_val = val[y0:, x0:x1]
        sample_tex = texture[y0:, x0:x1]
        sample_lab = lab_f[y0:, x0:x1].reshape(-1, 3)

        seed_mask = (
            (sample_sat <= np.percentile(sample_sat, 68))
            & (sample_tex <= np.percentile(sample_tex, 60))
            & (sample_val >= np.percentile(sample_val, 18))
            & (sample_val <= np.percentile(sample_val, 92))
        )

        if int(seed_mask.sum()) < max(100, (sample_sat.size // 50)):
            seed_mask = (
                (sample_sat <= np.percentile(sample_sat, 78))
                & (sample_tex <= np.percentile(sample_tex, 72))
                & (sample_val >= np.percentile(sample_val, 12))
            )

        seed_lab = lab_f[y0:, x0:x1][seed_mask]
        if seed_lab.size == 0:
            seed_lab = sample_lab

        center_lab = np.median(seed_lab.reshape(-1, 3), axis=0)
        lab_dev = np.median(np.abs(seed_lab.reshape(-1, 3) - center_lab), axis=0)
        lab_dev = np.maximum(lab_dev, np.array([8.0, 6.0, 6.0], dtype=np.float32))

        lab_dist = np.sqrt(np.sum(((lab_f - center_lab) / lab_dev) ** 2, axis=2))
        road_like = np.exp(-0.5 * lab_dist * lab_dist).astype(np.float32)

        sat_limit = float(max(28.0, np.percentile(sample_sat, 82)))
        tex_limit = float(max(18.0, np.percentile(sample_tex, 72)))
        val_low = float(max(8.0, np.percentile(sample_val, 6) - 10.0))
        val_high = float(min(255.0, np.percentile(sample_val, 96) + 12.0))

        sat_score = np.clip((sat_limit - sat) / max(sat_limit, 1.0), 0.0, 1.0)
        tex_score = np.clip((tex_limit - texture) / max(tex_limit, 1.0), 0.0, 1.0)
        val_gate = ((val >= val_low) & (val <= val_high)).astype(np.float32)

        confidence = 0.58 * road_like + 0.18 * sat_score + 0.16 * tex_score + 0.08 * val_gate
        confidence[: int(h * HORIZON_RATIO), :] = 0.0

        center_bias = np.ones((h, w), dtype=np.float32)
        xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
        # Keep right lane reachable, but suppress extreme right image border
        # where sidewalk often dominates under perspective.
        right_penalty = np.clip((xs - 0.84) / 0.16, 0.0, 1.0)
        lateral_bias = 1.0 - 0.12 * np.abs((xs - 0.56) / 0.56) - 0.42 * right_penalty * right_penalty
        confidence *= np.clip(lateral_bias[None, :], 0.45, 1.05)
        return np.clip(confidence, 0.0, 1.0)

    @staticmethod
    def _estimate_row_right_road_edge(
        left: int,
        right: int,
        conf_row: np.ndarray,
    ) -> int:
        if right <= left + MIN_ROAD_WIDTH_PX:
            return right
        row = conf_row[left:right + 1].astype(np.float32)
        if row.size < 10:
            return right

        sm = _smooth_1d(row, k=7)
        ref = float(np.percentile(sm, 58))
        min_idx = int(max(0, sm.size * 0.35))
        best = right
        good_run = 0
        for j in range(sm.size - 1, min_idx - 1, -1):
            if sm[j] >= 0.92 * ref:
                good_run += 1
                if good_run >= 4:
                    best = left + j
            else:
                if good_run >= 4:
                    break
                good_run = 0
        return int(np.clip(best, left + RIGHT_LANE_MIN_WIDTH_PX, right))

    def _extract_corridor_mask(self, confidence: np.ndarray, scene_cut: bool) -> np.ndarray:
        h, w = confidence.shape[:2]
        thr = float(max(0.42, np.percentile(confidence[int(h * 0.55):, :], 74)))
        candidate = (confidence >= thr).astype(np.uint8)

        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k7, iterations=2)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k5, iterations=1)

        support = self._trace_bottom_corridor(candidate, confidence, scene_cut)
        if support is None:
            support = self._fallback_bottom_component(candidate)
        support = self._finalize_mask(support)

        corridor = self._extract_right_lane_from_support(support, confidence, scene_cut)
        if corridor is None:
            corridor = self._fallback_right_lane(support)

        return self._finalize_mask(corridor)

    def _extract_right_lane_from_support(
        self,
        support: np.ndarray,
        confidence: np.ndarray,
        scene_cut: bool,
    ) -> np.ndarray | None:
        h, w = support.shape[:2]
        y_min = int(h * max(HORIZON_RATIO + 0.05, 0.40))

        rows_y: list[float] = []
        inner_xs: list[float] = []
        outer_xs: list[float] = []
        prev_outer_x: float | None = None
        prev_width: float | None = None

        for y in range(h - 1, y_min - 1, -4):
            xs = np.where(support[y] > 0)[0]
            if xs.size < MIN_ROAD_WIDTH_PX:
                continue

            left = int(xs[0])
            right = int(xs[-1])
            road_width = float(right - left + 1)
            lane_width = float(np.clip(
                road_width * RIGHT_LANE_WIDTH_FRACTION,
                RIGHT_LANE_MIN_WIDTH_PX,
                max(RIGHT_LANE_MIN_WIDTH_PX, road_width * RIGHT_LANE_MAX_WIDTH_RATIO),
            ))

            right_road_edge = self._estimate_row_right_road_edge(left, right, confidence[y])
            edge_inset = float(np.clip(
                road_width * RIGHT_EDGE_INSET_FRAC,
                RIGHT_EDGE_INSET_MIN_PX,
                RIGHT_EDGE_INSET_MAX_PX,
            ))
            outer_x = float(right_road_edge - max(float(RIGHT_EDGE_MARGIN_PX), edge_inset * 0.75))
            inner_x = float(max(left + 2, outer_x - lane_width))

            if prev_outer_x is not None:
                max_shift = max(7.0, 0.035 * w)
                outer_x = float(np.clip(outer_x, prev_outer_x - max_shift, prev_outer_x + max_shift))
                if prev_width is not None:
                    lane_width = float(np.clip(lane_width, prev_width - 8.0, prev_width + 12.0))
                inner_x = float(max(left + 2, outer_x - lane_width))

            min_lane_w = max(float(RIGHT_LANE_MIN_WIDTH_PX), float(road_width * INNER_MARKING_MIN_RATIO))
            max_lane_w = max(min_lane_w + 6.0, float(road_width * INNER_MARKING_MAX_RATIO))
            lane_width = float(np.clip(outer_x - inner_x, min_lane_w, max_lane_w))
            inner_x = float(max(left + 2, outer_x - lane_width))

            lane_slice_left = int(max(left, round(inner_x)))
            lane_slice_right = int(min(right, round(outer_x)))
            if lane_slice_right <= lane_slice_left + 3:
                continue

            lane_conf = float(np.mean(confidence[y, lane_slice_left:lane_slice_right + 1]))
            road_conf = float(np.mean(confidence[y, left:right + 1]))
            if lane_conf + 0.02 < road_conf:
                tighter_width = float(np.clip(
                    0.36 * road_width,
                    RIGHT_LANE_MIN_WIDTH_PX,
                    max(RIGHT_LANE_MIN_WIDTH_PX, 0.50 * road_width),
                ))
                inner_x = float(max(left + 2, outer_x - tighter_width))

            rows_y.append(float(y))
            inner_xs.append(float(inner_x))
            outer_xs.append(float(outer_x))
            prev_outer_x = float(outer_x)
            prev_width = float(outer_x - inner_x)

        if len(rows_y) < 6:
            return None

        ys_arr = np.asarray(rows_y, dtype=np.float32)[::-1]
        inner_arr = _smooth_1d(np.asarray(inner_xs, dtype=np.float32)[::-1], k=9)
        outer_arr = _smooth_1d(np.asarray(outer_xs, dtype=np.float32)[::-1], k=9)

        for i in range(1, len(ys_arr)):
            outer_arr[i] = np.clip(outer_arr[i], outer_arr[i - 1] - 10.0, outer_arr[i - 1] + 6.0)
            inner_arr[i] = np.clip(inner_arr[i], inner_arr[i - 1] - 12.0, inner_arr[i - 1] + 8.0)
            if outer_arr[i] <= inner_arr[i] + RIGHT_LANE_MIN_WIDTH_PX:
                inner_arr[i] = outer_arr[i] - RIGHT_LANE_MIN_WIDTH_PX

        polygon = np.vstack(
            [
                np.stack([outer_arr, ys_arr], axis=1),
                np.stack([inner_arr[::-1], ys_arr[::-1]], axis=1),
            ]
        ).astype(np.int32)
        if len(polygon) < 3 or abs(cv2.contourArea(polygon)) < 300:
            return None

        lane_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(lane_mask, [polygon], 1)
        lane_mask = (lane_mask & support.astype(np.uint8)).astype(np.uint8)
        lane_mask = cv2.morphologyEx(
            lane_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2,
        )
        return lane_mask

    def _fallback_right_lane(self, support: np.ndarray) -> np.ndarray:
        h, w = support.shape[:2]
        lane = np.zeros_like(support, dtype=np.uint8)
        for y in range(h):
            xs = np.where(support[y] > 0)[0]
            if xs.size < MIN_ROAD_WIDTH_PX:
                continue
            left = int(xs[0])
            right = int(xs[-1])
            road_width = float(right - left + 1)
            lane_width = int(np.clip(
                road_width * RIGHT_LANE_WIDTH_FRACTION,
                RIGHT_LANE_MIN_WIDTH_PX,
                max(RIGHT_LANE_MIN_WIDTH_PX, road_width * RIGHT_LANE_MAX_WIDTH_RATIO),
            ))
            start_x = max(left + 2, right - lane_width)
            end_x = max(start_x + 1, right - RIGHT_EDGE_MARGIN_PX)
            lane[y, start_x:end_x + 1] = 1
        return lane

    def _trace_bottom_corridor(
        self,
        candidate: np.ndarray,
        confidence: np.ndarray,
        scene_cut: bool,
    ) -> np.ndarray | None:
        h, w = candidate.shape[:2]
        y_min = int(h * max(HORIZON_RATIO + 0.04, 0.38))
        sample_step = 4

        bottom_band = candidate[int(h * 0.82):, :]
        if bottom_band.size == 0:
            return None

        col_strength = bottom_band.sum(axis=0).astype(np.float32)
        if np.max(col_strength) <= 0:
            return None

        x_lo = int(w * ANCHOR_MIN_X_RATIO)
        x_hi = int(w * ANCHOR_MAX_X_RATIO)
        if x_hi > x_lo + 8:
            gated = np.zeros_like(col_strength)
            gated[x_lo:x_hi] = col_strength[x_lo:x_hi]
            if np.max(gated) > 0:
                col_strength = gated

        if self.last_small_mask is not None and not scene_cut and self.last_small_mask.shape == candidate.shape:
            prev_band = self.last_small_mask[int(h * 0.82):, :]
            prev_cols = prev_band.sum(axis=0).astype(np.float32)
            if np.max(prev_cols) > 0:
                col_strength = 0.75 * col_strength + 0.25 * prev_cols

        center_x = float(np.argmax(col_strength))

        ys: list[float] = []
        lefts: list[float] = []
        rights: list[float] = []
        last_width: float | None = None

        for y in range(h - 1, y_min - 1, -sample_step):
            xs = np.where(candidate[y] > 0)[0]
            if xs.size == 0:
                continue

            runs = _find_runs(xs)
            scored: list[tuple[float, int, int, float]] = []
            for left, right in runs:
                width = right - left + 1
                if width < MIN_ROAD_WIDTH_PX:
                    continue
                mid = 0.5 * (left + right)
                dist = abs(mid - center_x)
                conf = float(np.mean(confidence[y, left:right + 1]))
                width_penalty = 0.0
                if last_width is not None:
                    width_penalty = abs(width - last_width) / max(last_width, 1.0)
                right_border_penalty = 0.0
                if right >= int(w * RIGHT_BORDER_PENALTY_RATIO):
                    right_border_penalty = 0.42 + 0.26 * ((right - w * RIGHT_BORDER_PENALTY_RATIO) / max(w * 0.05, 1.0))
                score = conf * 2.6 - dist / max(w, 1) - 0.35 * width_penalty - right_border_penalty
                scored.append((score, left, right, mid))

            if not scored:
                continue

            _, left, right, mid = max(scored, key=lambda item: item[0])
            width = float(right - left + 1)

            if last_width is not None:
                max_growth = max(8.0, 0.14 * w)
                min_width = max(float(MIN_ROAD_WIDTH_PX), last_width - max_growth)
                max_width = last_width + max_growth
                if width < min_width or width > max_width:
                    width = float(np.clip(width, min_width, max_width))
                    half = 0.5 * width
                    left = int(max(0, round(mid - half)))
                    right = int(min(w - 1, round(mid + half)))

            ys.append(float(y))
            lefts.append(float(left))
            rights.append(float(right))
            center_x = 0.72 * center_x + 0.28 * mid
            last_width = float(right - left + 1)

        if len(ys) < 6:
            return None

        ys_arr = np.asarray(ys, dtype=np.float32)[::-1]
        left_arr = _smooth_1d(np.asarray(lefts, dtype=np.float32)[::-1], k=11)
        right_arr = _smooth_1d(np.asarray(rights, dtype=np.float32)[::-1], k=11)

        for i in range(1, len(ys_arr)):
            left_arr[i] = np.clip(left_arr[i], left_arr[i - 1] - 7.0, left_arr[i - 1] + 12.0)
            right_arr[i] = np.clip(right_arr[i], right_arr[i - 1] - 12.0, right_arr[i - 1] + 7.0)
            width = right_arr[i] - left_arr[i]
            if width < MIN_ROAD_WIDTH_PX:
                pad = 0.5 * (MIN_ROAD_WIDTH_PX - width)
                left_arr[i] -= pad
                right_arr[i] += pad

        polygon = np.vstack(
            [
                np.stack([right_arr, ys_arr], axis=1),
                np.stack([left_arr[::-1], ys_arr[::-1]], axis=1),
            ]
        ).astype(np.int32)

        if len(polygon) < 3 or abs(cv2.contourArea(polygon)) < 500:
            return None

        corridor = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(corridor, [polygon], 1)
        corridor = (corridor & candidate).astype(np.uint8)
        return corridor

    def _fallback_bottom_component(self, candidate: np.ndarray) -> np.ndarray:
        h, w = candidate.shape[:2]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        if num_labels <= 1:
            return candidate

        best_label = 0
        best_score = -1.0
        bottom_gate = int(h * 0.86)
        center_x = 0.5 * w
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(200, int(0.003 * h * w)):
                continue
            left = int(stats[label, cv2.CC_STAT_LEFT])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            top = int(stats[label, cv2.CC_STAT_TOP])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            bottom = top + height
            if bottom < bottom_gate:
                continue
            comp_center = left + 0.5 * width
            dist = abs(comp_center - center_x) / max(w, 1)
            score = area * (1.0 + 0.8 * (bottom / max(h, 1))) * (1.1 - min(dist, 1.0))
            if score > best_score:
                best_score = score
                best_label = label

        if best_label == 0:
            return candidate
        return (labels == best_label).astype(np.uint8)

    def _finalize_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        h, _ = mask.shape[:2]
        mask[: int(h * HORIZON_RATIO), :] = 0

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            filtered = np.zeros_like(mask)
            min_area = max(250, int(mask.size * 0.004))
            for label in range(1, num_labels):
                area = int(stats[label, cv2.CC_STAT_AREA])
                top = int(stats[label, cv2.CC_STAT_TOP])
                height = int(stats[label, cv2.CC_STAT_HEIGHT])
                bottom = top + height
                if area >= min_area and bottom >= int(h * 0.84):
                    filtered[labels == label] = 1
            if np.any(filtered):
                mask = filtered
        return mask

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


def process_video(
    input_path: str,
    output_path: str,
    segmenter: RoadSegmenter,
    show_direction: bool = False,
    save_frames: bool = False,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Ошибка: не удается открыть {input_path}", file=sys.stderr)
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height * 2))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"\rОбработка кадра {frame_count}", end="", flush=True)

        mask = segmenter.get_road_mask(frame)
        direction = segmenter.estimate_road_direction(mask) if show_direction else None

        vis = frame.copy()
        vis[mask > 0] = [0, 255, 0]

        mask_vis = np.zeros((height, width, 3), dtype=np.uint8)
        mask_vis[mask > 0] = [0, 255, 0]
        combined = np.vstack([vis, mask_vis])

        if direction is not None:
            h_vis, w_vis = vis.shape[:2]
            center = (w_vis // 2, h_vis // 2)
            end = tuple((np.array(center) + 100 * direction).astype(int))
            cv2.arrowedLine(combined, center, end, (0, 0, 255), 3)

        out.write(combined)

        if save_frames and frame_count <= 10:
            cv2.imwrite(f"frame_{frame_count:04d}.png", combined)

    cap.release()
    out.release()
    print(f"\nГотово! Результат: {output_path}")


def process_image(input_path: str, segmenter: RoadSegmenter, show_direction: bool = False):
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Ошибка: не удается загрузить {input_path}", file=sys.stderr)
        return

    print(f"Обработка изображения {input_path}")

    mask = segmenter.get_road_mask(frame)
    direction = segmenter.estimate_road_direction(mask) if show_direction else None

    vis = frame.copy()
    vis[mask > 0] = [0, 255, 0]

    h, _ = mask.shape[:2]
    centers = []
    for y in [h // 4, h // 2, 3 * h // 4]:
        center_x = segmenter.road_center_x_at_y(mask, y)
        if center_x is not None:
            centers.append((int(center_x), y))
            cv2.circle(vis, (int(center_x), y), 5, (255, 0, 0), -1)

    if direction is not None:
        h_vis, w_vis = vis.shape[:2]
        center = (w_vis // 2, h_vis // 2)
        end = tuple((np.array(center) + 100 * direction).astype(int))
        cv2.arrowedLine(vis, center, end, (0, 0, 255), 3)

    base_name = Path(input_path).stem
    output_path = f"{base_name}_road.png"
    cv2.imwrite(output_path, vis)

    print(f"Результат сохранен: {output_path}")
    road_pixels = np.sum(mask > 0)
    print(f"Пикселей дороги: {road_pixels:,} ({road_pixels / (h * mask.shape[1]) * 100:.1f}%)")
    if centers:
        print("Центры дороги по высоте:", centers)
    if direction is not None:
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        print(f"Направление дороги: {direction}, угол: {angle:.1f}°")


def main():
    parser = argparse.ArgumentParser(description="Тестирование RoadSegmenter")
    parser.add_argument("input", help="Входной файл (видео или изображение)")
    parser.add_argument("-o", "--output", help="Выходной файл (только для видео)")
    parser.add_argument(
        "--update-every",
        type=int,
        default=1,
        help="Обновлять маску каждые N кадров",
    )
    parser.add_argument(
        "--downscale",
        type=float,
        default=0.5,
        help="Масштаб для ускорения",
    )
    parser.add_argument(
        "--direction",
        action="store_true",
        help="Показывать направление дороги",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Сохранять первые 10 кадров",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Живое тестирование с веб-камеры",
    )

    args = parser.parse_args()

    if args.live:
        segmenter = RoadSegmenter(args.update_every, args.downscale)
        cap = cv2.VideoCapture(0)

        print("Нажмите 'q' для выхода, 's' для сохранения кадра")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            mask = segmenter.get_road_mask(frame)
            direction = segmenter.estimate_road_direction(mask) if args.direction else None

            vis = frame.copy()
            vis[mask > 0] = [0, 255, 0]

            if direction is not None:
                h_vis, w_vis = vis.shape[:2]
                center = (w_vis // 2, h_vis // 2)
                end = tuple((np.array(center) + 100 * direction).astype(int))
                cv2.arrowedLine(vis, center, end, (0, 0, 255), 3)

            cv2.putText(
                vis,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            cv2.imshow("RoadSegmenter Live", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                cv2.imwrite(f"live_frame_{frame_count:04d}.png", vis)
                print(f"Сохранен кадр {frame_count}")

        cap.release()
        cv2.destroyAllWindows()
        return

    if not Path(args.input).exists():
        print(f"Ошибка: файл {args.input} не найден", file=sys.stderr)
        return

    segmenter = RoadSegmenter(args.update_every, args.downscale)
    if args.input.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        output_path = args.output or f"{Path(args.input).stem}_road.mp4"
        process_video(args.input, output_path, segmenter, args.direction, args.save_frames)
    else:
        process_image(args.input, segmenter, args.direction)


if __name__ == "__main__":
    main()
