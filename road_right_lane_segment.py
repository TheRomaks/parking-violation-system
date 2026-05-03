import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

# Conservative thresholds so we do not leak onto sidewalks.
ROAD_PROB_THRESHOLD = 0.30
ROAD_SAFE_THRESHOLD = 0.38
ROAD_SIDEWALK_MARGIN = 0.05
SAFE_SIDEWALK_MARGIN = 0.10
LANE_MARK_THRESHOLD = 0.30
MIN_COMPONENT_AREA = 700
BEV_MIN_Y_RATIO = 0.20
BEV_SAMPLE_STEP = 8
OUTER_SEARCH_BAND = 80
DIVIDER_SEARCH_BAND = 110
MIN_LANE_WIDTH_PX = 20
MIN_ROAD_WIDTH_PX = 36
PREDICT_KEEP_FRAMES = 12
RESET_AFTER_LOST = 20


# ----------------------------- Model loading -----------------------------


def load_model(device: str):
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    road_idx = None
    sidewalk_idx = None
    for idx, label in model.config.id2label.items():
        label_l = str(label).lower()
        if label_l == "road" or "road" in label_l:
            road_idx = int(idx)
        if label_l == "sidewalk" or "sidewalk" in label_l:
            sidewalk_idx = int(idx)

    if road_idx is None:
        raise RuntimeError(f"Не найден класс road. Метки: {model.config.id2label}")
    if sidewalk_idx is None:
        sidewalk_idx = road_idx

    return processor, model, road_idx, sidewalk_idx


@torch.no_grad()
def infer_prob_maps(frame_bgr: np.ndarray, processor, model, device: str) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    logits = F.interpolate(out.logits, size=frame_bgr.shape[:2], mode="bilinear", align_corners=False)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return probs


# ----------------------------- Mask utilities -----------------------------


def largest_component(mask: np.ndarray, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray:
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    h = mask.shape[0]
    best_lbl = 0
    best_score = -1.0
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        ys = np.where(labels == lbl)[0]
        if ys.size == 0:
            continue
        score = float(area) * (1.0 + ys.max() / max(h - 1, 1))
        if score > best_score:
            best_score = score
            best_lbl = lbl

    if best_lbl == 0:
        return mask
    return (labels == best_lbl).astype(np.uint8)


def make_road_masks(prob_maps: np.ndarray, road_idx: int, sidewalk_idx: int):
    road_prob = prob_maps[road_idx]
    sidewalk_prob = prob_maps[sidewalk_idx]

    support = ((road_prob >= ROAD_PROB_THRESHOLD) & (road_prob >= sidewalk_prob + ROAD_SIDEWALK_MARGIN)).astype(np.uint8)
    safe = ((road_prob >= ROAD_SAFE_THRESHOLD) & (road_prob >= sidewalk_prob + SAFE_SIDEWALK_MARGIN)).astype(np.uint8)

    k = np.ones((7, 7), np.uint8)
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, k, iterations=2)
    support = cv2.morphologyEx(support, cv2.MORPH_OPEN, k, iterations=1)
    support = largest_component(support)
    support = regularize_corridor_mask(support)

    safe = cv2.morphologyEx(safe, cv2.MORPH_CLOSE, k, iterations=2)
    safe = cv2.morphologyEx(safe, cv2.MORPH_OPEN, k, iterations=1)
    safe = largest_component(safe)
    safe = regularize_corridor_mask(safe)

    # Keep the safe mask inside the broader road support.
    safe = (safe & support).astype(np.uint8)

    # A geometry mask that is stricter than support but looser than safe.
    geo = ((road_prob >= ROAD_PROB_THRESHOLD) & (road_prob >= sidewalk_prob + 0.03)).astype(np.uint8)
    geo = cv2.morphologyEx(geo, cv2.MORPH_CLOSE, k, iterations=1)
    geo = largest_component(geo)
    geo = regularize_corridor_mask(geo)
    geo = (geo & support).astype(np.uint8)

    return road_prob, sidewalk_prob, support, safe, geo


# ----------------------------- Geometry helpers -----------------------------


def smooth_1d(arr: np.ndarray, k: int = 7) -> np.ndarray:
    if arr is None or len(arr) < 3:
        return arr
    k = max(3, int(k) | 1)
    pad = k // 2
    padded = np.pad(arr.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(padded, kernel, mode="valid")


def fit_line_x_of_y(ys: np.ndarray, xs: np.ndarray) -> Tuple[float, float]:
    A = np.vstack([ys, np.ones_like(ys)]).T
    sol, _, _, _ = np.linalg.lstsq(A, xs, rcond=None)
    return float(sol[0]), float(sol[1])


def resample_contour(points: np.ndarray, n: int = 60) -> Optional[np.ndarray]:
    if points is None or len(points) < 3:
        return None
    pts = points.astype(np.float32)
    closed = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if float(seg.sum()) < 1e-6:
        return None
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, cum[-1], n, endpoint=False)
    out = []
    for ti in t:
        idx = np.searchsorted(cum, ti, side="right") - 1
        idx = np.clip(idx, 0, len(closed) - 2)
        denom = max(cum[idx + 1] - cum[idx], 1e-6)
        a = (ti - cum[idx]) / denom
        p = (1.0 - a) * closed[idx] + a * closed[idx + 1]
        out.append(p)
    return np.asarray(out, dtype=np.float32)


def interp_nan_curve(ys: np.ndarray, xs: np.ndarray) -> Optional[np.ndarray]:
    valid = np.isfinite(xs)
    if valid.sum() < 4:
        return None
    ys_v = ys[valid]
    xs_v = xs[valid]
    order = np.argsort(ys_v)
    ys_v = ys_v[order]
    xs_v = xs_v[order]
    # Fill with nearest valid values on both ends.
    full = np.interp(ys, ys_v, xs_v, left=xs_v[0], right=xs_v[-1])
    return full.astype(np.float32)


def row_bounds(mask: np.ndarray, y: int) -> Optional[Tuple[int, int]]:
    xs = np.where(mask[y] > 0)[0]
    if xs.size < 2:
        return None
    return int(xs[0]), int(xs[-1])


def regularize_corridor_mask(mask: np.ndarray, y_min_ratio: float = BEV_MIN_Y_RATIO) -> np.ndarray:
    """
    Turn a noisy mask into one smooth drivable corridor.
    This suppresses right-edge sidewalk leakage and fills small row gaps.
    """
    h, w = mask.shape[:2]
    ys = []
    lefts = []
    rights = []

    for y in range(h - 1, int(h * y_min_ratio) - 1, -4):
        rb = row_bounds(mask, y)
        if rb is None:
            continue
        l, r = rb
        if r - l < MIN_ROAD_WIDTH_PX:
            continue
        ys.append(float(y))
        lefts.append(float(l))
        rights.append(float(r))

    if len(ys) < 6:
        return mask.astype(np.uint8)

    ys = np.asarray(ys, dtype=np.float32)[::-1]
    lefts = smooth_1d(np.asarray(lefts, dtype=np.float32)[::-1], k=11)
    rights = smooth_1d(np.asarray(rights, dtype=np.float32)[::-1], k=11)

    widths = rights - lefts
    med_w = float(np.median(widths))
    min_w = max(float(MIN_ROAD_WIDTH_PX), 0.55 * med_w)

    valid = widths >= min_w
    if int(valid.sum()) < 5:
        return mask.astype(np.uint8)

    ys = ys[valid]
    lefts = lefts[valid]
    rights = rights[valid]

    max_step = max(8.0, 0.045 * w)
    for i in range(1, len(ys)):
        lefts[i] = max(lefts[i], lefts[i - 1] - max_step)
        lefts[i] = min(lefts[i], lefts[i - 1] + max_step)
        rights[i] = max(rights[i], rights[i - 1] - max_step)
        rights[i] = min(rights[i], rights[i - 1] + max_step)

    poly = np.vstack([
        np.stack([rights, ys], axis=1),
        np.stack([lefts[::-1], ys[::-1]], axis=1),
    ]).astype(np.int32)

    corridor = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(corridor, [poly], 1)
    corridor = cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    corridor = (corridor & mask.astype(np.uint8)).astype(np.uint8)
    corridor = largest_component(corridor, min_area=max(MIN_COMPONENT_AREA // 2, 300))
    return corridor


# ----------------------------- Lane state -----------------------------


@dataclass
class TrackState:
    prev_quad: Optional[np.ndarray] = None
    prev_outer_curve: Optional[np.ndarray] = None
    prev_div_curve: Optional[np.ndarray] = None
    prev_poly: Optional[np.ndarray] = None
    prev_lane_fraction: float = 0.45
    lost_frames: int = 0


# ----------------------------- Road quad estimation -----------------------------


def estimate_road_quad_from_mask(mask: np.ndarray, state: TrackState) -> Optional[np.ndarray]:
    h, w = mask.shape[:2]
    ys, lefts, rights = [], [], []

    y_min = int(h * BEV_MIN_Y_RATIO)
    for y in range(h - 1, y_min - 1, -6):
        rb = row_bounds(mask, y)
        if rb is None:
            continue
        l, r = rb
        if r - l < MIN_ROAD_WIDTH_PX:
            continue
        ys.append(y)
        lefts.append(l)
        rights.append(r)

    if len(ys) < 5:
        return state.prev_quad.copy().astype(np.float32) if state.prev_quad is not None else None

    ys = np.asarray(ys, dtype=np.float32)
    lefts = smooth_1d(np.asarray(lefts, dtype=np.float32), k=7)
    rights = smooth_1d(np.asarray(rights, dtype=np.float32), k=7)

    l_slope, l_intercept = fit_line_x_of_y(ys, lefts)
    r_slope, r_intercept = fit_line_x_of_y(ys, rights)

    bottom_y = float(h - 1)
    top_y = float(np.clip(np.percentile(ys, 25), h * 0.18, h * 0.60))

    bl = np.array([l_slope * bottom_y + l_intercept, bottom_y], dtype=np.float32)
    br = np.array([r_slope * bottom_y + r_intercept, bottom_y], dtype=np.float32)
    tl = np.array([l_slope * top_y + l_intercept, top_y], dtype=np.float32)
    tr = np.array([r_slope * top_y + r_intercept, top_y], dtype=np.float32)

    pad = max(4, int(0.006 * w))
    bl[0] -= pad
    tl[0] -= pad
    br[0] += pad
    tr[0] += pad

    quad = np.array([bl, br, tr, tl], dtype=np.float32)
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)

    if abs(cv2.contourArea(quad.astype(np.int32))) < 1200:
        return state.prev_quad.copy().astype(np.float32) if state.prev_quad is not None else None

    return quad


# ----------------------------- Bird's-eye transform -----------------------------


def order_quad_points(quad: np.ndarray) -> np.ndarray:
    pts = quad.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([bl, br, tr, tl], dtype=np.float32)


def warp_to_birds_eye(frame: np.ndarray, mask: np.ndarray, quad: np.ndarray):
    quad = order_quad_points(quad)
    bottom_width = np.linalg.norm(quad[1] - quad[0])
    top_width = np.linalg.norm(quad[2] - quad[3])

    out_w = int(max(320, bottom_width * 1.05))
    out_h = int(max(420, max(bottom_width, top_width) * 1.7))

    dst = np.array([
        [0, out_h - 1],
        [out_w - 1, out_h - 1],
        [out_w - 1, 0],
        [0, 0],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    Minv = cv2.getPerspectiveTransform(dst, quad)

    warped_frame = cv2.warpPerspective(frame, M, (out_w, out_h))
    warped_mask = cv2.warpPerspective((mask * 255).astype(np.uint8), M, (out_w, out_h))
    warped_mask = (warped_mask > 0).astype(np.uint8)
    return warped_frame, warped_mask, M, Minv


def warp_prob_map(prob_map: np.ndarray, M: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    out_w, out_h = out_size
    warped = cv2.warpPerspective(prob_map.astype(np.float32), M, (out_w, out_h))
    return warped.astype(np.float32)


# ----------------------------- Lane marking + boundaries in BEV -----------------------------


def lane_mark_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 170), (180, 65, 255))
    yellow = cv2.inRange(hsv, (15, 55, 70), (40, 255, 255))
    mask = cv2.bitwise_or(white, yellow)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return (mask > 0).astype(np.uint8)


def detect_outer_edge_curve_in_bev(road_prob: np.ndarray, sidewalk_prob: np.ndarray, geom_mask: np.ndarray, state: TrackState):
    h, w = road_prob.shape[:2]
    y_grid = np.arange(h - 1, int(h * BEV_MIN_Y_RATIO) - 1, -BEV_SAMPLE_STEP, dtype=np.int32)

    prev_curve = state.prev_outer_curve
    prev_pred = None
    if prev_curve is not None and len(prev_curve) == len(y_grid):
        prev_pred = prev_curve

    xs = np.full(len(y_grid), np.nan, dtype=np.float32)
    road_score = road_prob - sidewalk_prob

    for i, y in enumerate(y_grid):
        rb = row_bounds(geom_mask, int(y))
        if rb is None:
            continue
        left, right = rb
        if right - left < MIN_ROAD_WIDTH_PX:
            continue

        x_min = max(left + 5, int(left + 0.25 * (right - left)))
        x_max = right

        if prev_pred is not None:
            p = int(np.clip(prev_pred[i], left + 5, right))
            x_min = max(x_min, p - OUTER_SEARCH_BAND)
            x_max = min(x_max, p + OUTER_SEARCH_BAND)

        if x_max <= x_min + 4:
            continue

        candidates = np.arange(x_min, x_max + 1, dtype=np.int32)
        if candidates.size == 0:
            continue

        cand_scores = np.full(candidates.shape[0], -1e6, dtype=np.float32)
        for j, x in enumerate(candidates):
            inside0 = max(left, x - 6)
            inside1 = x + 1
            outside0 = x + 1
            outside1 = min(w, x + 8)
            if inside1 <= inside0 or outside1 <= outside0:
                continue

            inside = float(np.mean(road_score[int(y), inside0:inside1]))
            outside = float(np.mean(road_score[int(y), outside0:outside1]))
            geom_bonus = 0.08 if geom_mask[int(y), x] > 0 else -0.10
            cand_scores[j] = inside - outside + geom_bonus

        best_j = int(np.argmax(cand_scores))
        best_x = int(candidates[best_j])
        best_score = float(cand_scores[best_j])

        if best_score > 0.02:
            xs[i] = float(best_x)
        elif prev_pred is not None:
            xs[i] = float(np.clip(prev_pred[i], left + 5, right))
        else:
            xs[i] = float(right)

    curve = interp_nan_curve(y_grid.astype(np.float32), xs)
    if curve is None:
        raw = []
        for y in y_grid:
            rb = row_bounds(geom_mask, int(y))
            raw.append(float(rb[1]) if rb is not None else np.nan)
        curve = interp_nan_curve(y_grid.astype(np.float32), np.asarray(raw, dtype=np.float32))
        if curve is None:
            return None

    curve = smooth_1d(curve, k=9)

    if prev_pred is not None:
        if len(prev_pred) == len(curve):
            curve = 0.88 * prev_pred + 0.12 * curve

    x_bottom = float(curve[0])
    if not (0.20 * w <= x_bottom <= 0.995 * w):
        return None

    return y_grid.astype(np.float32), curve.astype(np.float32)


def detect_divider_curve_in_bev(
    warped_frame: np.ndarray,
    road_prob: np.ndarray,
    sidewalk_prob: np.ndarray,
    geom_mask: np.ndarray,
    outer_curve: Tuple[np.ndarray, np.ndarray],
    state: TrackState,
):
    h, w = warped_frame.shape[:2]
    y_grid, outer_xs = outer_curve
    marks = lane_mark_mask(warped_frame)
    marks = (marks * geom_mask).astype(np.uint8)

    prev_curve = state.prev_div_curve
    prev_pred = None
    if prev_curve is not None and len(prev_curve) == len(y_grid):
        prev_pred = prev_curve

    xs = np.full(len(y_grid), np.nan, dtype=np.float32)
    road_score = road_prob - sidewalk_prob

    for i, y in enumerate(y_grid):
        rb = row_bounds(geom_mask, int(y))
        if rb is None:
            continue
        left, right = rb
        if right - left < MIN_ROAD_WIDTH_PX:
            continue

        outer_x = float(np.clip(outer_xs[i], left + MIN_LANE_WIDTH_PX + 4, right))
        road_w = max(outer_x - left, float(MIN_ROAD_WIDTH_PX))
        fallback_lane_fraction = float(np.clip(state.prev_lane_fraction, 0.28, 0.65))
        fallback_div = outer_x - fallback_lane_fraction * road_w

        x_lo = int(max(left + 6, left + 0.14 * road_w))
        x_hi = int(min(outer_x - MIN_LANE_WIDTH_PX, outer_x - 5))

        if prev_pred is not None:
            p = int(np.clip(prev_pred[i], x_lo, max(x_lo + 1, x_hi)))
            x_lo = max(x_lo, p - DIVIDER_SEARCH_BAND)
            x_hi = min(x_hi, p + DIVIDER_SEARCH_BAND)

        if x_hi <= x_lo + 4:
            xs[i] = float(np.clip(fallback_div, left + 5, outer_x - MIN_LANE_WIDTH_PX))
            continue

        row = cv2.GaussianBlur(marks[int(y)].astype(np.float32).reshape(1, -1), (1, 21), 0).reshape(-1)
        segment = row[x_lo:x_hi + 1]
        if segment.size > 0:
            peak_idx = int(np.argmax(segment))
            peak_val = float(segment[peak_idx])
        else:
            peak_val = 0.0
            peak_idx = 0

        candidate = None
        if peak_val >= LANE_MARK_THRESHOLD:
            candidate = float(x_lo + peak_idx)

        if candidate is None:
            # Secondary cue: divider is usually a local drop in road-to-sidewalk score near the middle-right of the road.
            score_seg = road_score[int(y), x_lo:x_hi + 1]
            if score_seg.size > 0:
                score_norm = (score_seg - np.min(score_seg)) / (np.ptp(score_seg) + 1e-6)
                if score_norm.max() > 0.55:
                    candidate = float(x_lo + int(np.argmax(score_norm)))

        if candidate is None:
            candidate = float(np.clip(fallback_div, left + 5, outer_x - MIN_LANE_WIDTH_PX))

        candidate = float(np.clip(candidate, left + 5, outer_x - MIN_LANE_WIDTH_PX))
        xs[i] = candidate

    curve = interp_nan_curve(y_grid.astype(np.float32), xs)
    if curve is None:
        fallback_xs = np.full(len(y_grid), np.nan, dtype=np.float32)
        for i, y in enumerate(y_grid):
            rb = row_bounds(geom_mask, int(y))
            if rb is None:
                continue
            left, right = rb
            road_w = max(float(right - left), float(MIN_ROAD_WIDTH_PX))
            lane_fraction = float(np.clip(state.prev_lane_fraction, 0.28, 0.65))
            fallback_xs[i] = float(np.clip(right - lane_fraction * road_w, left + 5, right - MIN_LANE_WIDTH_PX))
        curve = interp_nan_curve(y_grid.astype(np.float32), fallback_xs)
        if curve is None:
            return None

    curve = smooth_1d(curve, k=9)
    if prev_pred is not None and len(prev_pred) == len(curve):
        curve = 0.88 * prev_pred + 0.12 * curve

    # Update lane fraction estimate from observed geometry.
    frac_samples = []
    for i, y in enumerate(y_grid):
        rb = row_bounds(geom_mask, int(y))
        if rb is None:
            continue
        left, right = rb
        outer_x = float(np.clip(outer_xs[i], left + MIN_LANE_WIDTH_PX + 2, right))
        road_w = max(outer_x - left, 1.0)
        frac = (outer_x - curve[i]) / road_w
        if 0.15 <= frac <= 0.80:
            frac_samples.append(frac)

    if frac_samples:
        obs = float(np.median(frac_samples))
        state.prev_lane_fraction = 0.90 * state.prev_lane_fraction + 0.10 * np.clip(obs, 0.25, 0.70)

    min_gap_samples = []
    for i, y in enumerate(y_grid):
        rb = row_bounds(geom_mask, int(y))
        if rb is None:
            continue
        left, right = rb
        outer_x = float(np.clip(outer_xs[i], left + MIN_LANE_WIDTH_PX + 2, right))
        gap = outer_x - curve[i]
        if gap > 0:
            min_gap_samples.append(gap)

    if len(min_gap_samples) < 4:
        return None

    x_bottom = float(curve[0])
    if not (0.05 * w <= x_bottom <= 0.99 * w):
        return None

    return y_grid.astype(np.float32), curve.astype(np.float32)


def build_rightmost_lane_polygon_in_bev(
    warped_frame: np.ndarray,
    warped_road_prob: np.ndarray,
    warped_sidewalk_prob: np.ndarray,
    geom_mask: np.ndarray,
    state: TrackState,
) -> Optional[np.ndarray]:
    outer_curve = detect_outer_edge_curve_in_bev(warped_road_prob, warped_sidewalk_prob, geom_mask, state)
    if outer_curve is None:
        return None

    divider_curve = detect_divider_curve_in_bev(
        warped_frame,
        warped_road_prob,
        warped_sidewalk_prob,
        geom_mask,
        outer_curve,
        state,
    )
    if divider_curve is None:
        return None

    y_grid, outer_xs = outer_curve
    _, divider_xs = divider_curve

    # Update state curves for temporal smoothing.
    if state.prev_outer_curve is None or len(state.prev_outer_curve) != len(outer_xs):
        state.prev_outer_curve = outer_xs.copy()
    else:
        state.prev_outer_curve = 0.88 * state.prev_outer_curve + 0.12 * outer_xs

    if state.prev_div_curve is None or len(state.prev_div_curve) != len(divider_xs):
        state.prev_div_curve = divider_xs.copy()
    else:
        state.prev_div_curve = 0.88 * state.prev_div_curve + 0.12 * divider_xs

    outer_xs = state.prev_outer_curve.copy()
    divider_xs = state.prev_div_curve.copy()

    # Hard safety clamp: lane polygon cannot cross outside the curb-aware geometry mask.
    ys_int = y_grid.astype(np.int32)
    safe_outer = []
    safe_div = []
    for i, y in enumerate(ys_int):
        rb = row_bounds(geom_mask, int(y))
        if rb is None:
            continue
        left, right = rb
        outer_x = float(np.clip(outer_xs[i], left + MIN_LANE_WIDTH_PX + 2, right))
        divider_x = float(np.clip(divider_xs[i], left + 5, outer_x - MIN_LANE_WIDTH_PX))
        if outer_x <= divider_x + 5:
            continue
        safe_outer.append([outer_x, float(y)])
        safe_div.append([divider_x, float(y)])

    if len(safe_outer) < 5 or len(safe_div) < 5:
        return None

    safe_outer = np.asarray(safe_outer, dtype=np.float32)
    safe_div = np.asarray(safe_div, dtype=np.float32)

    poly = np.vstack([
        safe_outer,
        safe_div[::-1],
    ]).astype(np.int32)

    if len(poly) < 3 or abs(cv2.contourArea(poly)) < 200:
        return None
    return poly


# ----------------------------- Projection and hard clipping -----------------------------


def unwarp_polygon(polygon_warp: np.ndarray, Minv: np.ndarray) -> np.ndarray:
    pts = polygon_warp.reshape(-1, 1, 2).astype(np.float32)
    unwarped = cv2.perspectiveTransform(pts, Minv).reshape(-1, 2)
    return np.round(unwarped).astype(np.int32)


def clip_polygon_to_mask_by_rows(frame_shape, polygon: np.ndarray, mask: np.ndarray, vertical_step: int = 2) -> Optional[np.ndarray]:
    h, w = frame_shape[:2]
    if polygon is None or len(polygon) < 3:
        return None

    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [polygon.astype(np.int32)], 1)
    inter = (poly_mask & mask.astype(np.uint8)).astype(np.uint8)
    if inter.sum() < 100:
        return None

    ys = []
    lefts = []
    rights = []
    for y in range(0, h, vertical_step):
        xs = np.where(inter[y] > 0)[0]
        if xs.size < 2:
            continue
        ys.append(float(y))
        lefts.append(float(xs[0]))
        rights.append(float(xs[-1]))

    if len(ys) < 4:
        inter_u8 = (inter * 255).astype(np.uint8)
        contours, _ = cv2.findContours(inter_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 120:
            return None
        return cnt.reshape(-1, 2)

    ys = np.asarray(ys, dtype=np.float32)
    lefts = smooth_1d(np.asarray(lefts, dtype=np.float32), k=7)
    rights = smooth_1d(np.asarray(rights, dtype=np.float32), k=7)

    polygon2 = np.vstack([
        np.stack([rights, ys], axis=1),
        np.stack([lefts[::-1], ys[::-1]], axis=1),
    ]).astype(np.int32)

    if len(polygon2) < 3:
        return None
    return polygon2


# ----------------------------- Smoothing -----------------------------


def smooth_polygon(prev_poly: Optional[np.ndarray], curr_poly: np.ndarray, alpha: float = 0.88) -> np.ndarray:
    if prev_poly is None or len(prev_poly) < 3 or len(curr_poly) < 3:
        return curr_poly

    n = 60
    prev_rs = resample_contour(prev_poly, n)
    curr_rs = resample_contour(curr_poly, n)
    if prev_rs is None or curr_rs is None:
        return curr_poly

    dist = np.mean(np.linalg.norm(prev_rs - curr_rs, axis=1))
    scale = max(np.ptp(curr_rs[:, 0]) + np.ptp(curr_rs[:, 1]), 1.0)
    if dist > 0.28 * scale:
        return curr_poly

    blended = alpha * prev_rs + (1 - alpha) * curr_rs
    return np.round(blended).astype(np.int32)


# ----------------------------- Drawing -----------------------------


def draw_result(frame: np.ndarray, road_mask: np.ndarray, safe_mask: Optional[np.ndarray], polygon: Optional[np.ndarray], locked: bool) -> np.ndarray:
    out = frame.copy()

    road_vis = np.zeros_like(frame)
    road_vis[road_mask > 0] = (0, 180, 0)
    out = cv2.addWeighted(out, 1.0, road_vis, 0.16, 0)

    if safe_mask is not None:
        safe_vis = np.zeros_like(frame)
        safe_vis[safe_mask > 0] = (180, 80, 0)
        out = cv2.addWeighted(out, 1.0, safe_vis, 0.12, 0)

    if polygon is not None and len(polygon) >= 3:
        fill = out.copy()
        cv2.fillPoly(fill, [polygon.astype(np.int32)], (0, 255, 255))
        out = cv2.addWeighted(out, 1.0, fill, 0.30, 0)
        cv2.polylines(out, [polygon.astype(np.int32)], True, (0, 0, 255), 3)

    label = "Lane: locked" if locked else "Lane: not locked"
    color = (0, 255, 0) if locked else (0, 0, 255)
    cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return out


# ----------------------------- Pipeline -----------------------------


def process_video(input_path: str, output_path: str, device: str, debug_masks: bool = False):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Видео не найдено: {input_path}")

    processor, model, road_idx, sidewalk_idx = load_model(device)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать выходной файл: {output_path}")

    mask_writer = None
    if debug_masks:
        mask_path = os.path.splitext(output_path)[0] + "_mask.mp4"
        mask_writer = cv2.VideoWriter(mask_path, fourcc, fps, (width, height))

    state = TrackState()
    frame_id = 0
    print("Старт обработки...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        probs = infer_prob_maps(frame, processor, model, device)
        road_prob, sidewalk_prob, road_support, road_safe, road_geo = make_road_masks(probs, road_idx, sidewalk_idx)

        # Use geometry mask for road quad estimation so we avoid sidewalks.
        quad = estimate_road_quad_from_mask(road_geo, state)
        final_poly = None
        locked = False

        if quad is not None:
            if state.prev_quad is None:
                state.prev_quad = quad.copy().astype(np.float32)
            else:
                state.prev_quad = 0.90 * state.prev_quad + 0.10 * quad.astype(np.float32)

            quad_use = state.prev_quad.copy().astype(np.float32)
            warped_frame, warped_support, M, Minv = warp_to_birds_eye(frame, road_geo, quad_use)

            warped_road_prob = warp_prob_map(road_prob, M, (warped_frame.shape[1], warped_frame.shape[0]))
            warped_sidewalk_prob = warp_prob_map(sidewalk_prob, M, (warped_frame.shape[1], warped_frame.shape[0]))
            warped_geo = warp_prob_map(road_geo.astype(np.float32), M, (warped_frame.shape[1], warped_frame.shape[0]))
            warped_geo = (warped_geo > 0.5).astype(np.uint8)

            poly_warp = build_rightmost_lane_polygon_in_bev(
                warped_frame,
                warped_road_prob,
                warped_sidewalk_prob,
                warped_geo,
                state,
            )

            if poly_warp is not None:
                poly_unwarped = unwarp_polygon(poly_warp, Minv)
                clipped = clip_polygon_to_mask_by_rows(frame.shape, poly_unwarped, road_geo)
                if clipped is not None:
                    if state.prev_poly is not None:
                        clipped = smooth_polygon(state.prev_poly, clipped, alpha=0.90)
                        reclip = clip_polygon_to_mask_by_rows(frame.shape, clipped, road_geo)
                        if reclip is not None:
                            clipped = reclip
                    state.prev_poly = clipped.copy()
                    final_poly = clipped
                    locked = True
                    state.lost_frames = 0
                else:
                    state.lost_frames += 1
            else:
                state.lost_frames += 1
        else:
            state.lost_frames += 1

        # Keep the last stable polygon for a short while.
        if final_poly is None and state.prev_poly is not None and state.lost_frames <= PREDICT_KEEP_FRAMES:
            final_poly = state.prev_poly.copy()
            locked = True

        if state.lost_frames > RESET_AFTER_LOST:
            state.prev_poly = None
            state.prev_quad = None
            state.prev_outer_curve = None
            state.prev_div_curve = None
            state.prev_lane_fraction = 0.45
            state.lost_frames = 0

        result = draw_result(frame, road_geo, road_safe, final_poly, locked)
        cv2.putText(result, f"Frame: {frame_id}", (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(result)

        if mask_writer is not None:
            mask_bgr = cv2.cvtColor((road_geo * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            if final_poly is not None and len(final_poly) >= 3:
                cv2.fillPoly(mask_bgr, [final_poly.astype(np.int32)], (0, 255, 255))
                cv2.polylines(mask_bgr, [final_poly.astype(np.int32)], True, (0, 0, 255), 2)
            mask_writer.write(mask_bgr)

        frame_id += 1
        if frame_id % 30 == 0:
            print(f"Обработано кадров: {frame_id}")

    cap.release()
    writer.release()
    if mask_writer is not None:
        mask_writer.release()

    print(f"Готово: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Путь к входному видео")
    parser.add_argument("--output", required=True, help="Путь к выходному видео")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug-masks", action="store_true", help="Сохранять видео с маской")
    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        device=args.device,
        debug_masks=args.debug_masks,
    )


if __name__ == "__main__":
    main()
