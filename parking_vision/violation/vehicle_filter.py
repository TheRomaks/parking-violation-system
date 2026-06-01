from dataclasses import dataclass

import cv2
import numpy as np

from perception_types import Detection

from .geometry import bbox_intersection_area
from .types import SignZone


@dataclass(slots=True)
class VehicleCandidateFilter:
    min_height_ratio: float = 0.055
    min_area_ratio: float = 0.0012
    min_road_support: float = 0.08
    min_zone_support: float = 0.15
    edge_margin_px: int = 3
    edge_max_width_ratio: float = 0.35
    occlusion_overlap_threshold: float = 0.68

    def filter(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int] | tuple[int, int, int],
        road_mask: np.ndarray | None,
        zones: list[SignZone],
    ) -> list[Detection]:
        frame_h, frame_w = frame_shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            return []

        tolerant_road = self._dilate_mask(road_mask)
        scored: list[tuple[Detection, float]] = []

        for detection in detections:
            score = self._candidate_score(detection, frame_w, frame_h, tolerant_road, zones)
            if score is None:
                detection.metadata["candidate_filter"] = "rejected"
                continue
            detection.metadata["candidate_filter"] = "accepted"
            scored.append((detection, score))

        return self._suppress_occluded(scored)

    def _candidate_score(
        self,
        detection: Detection,
        frame_w: int,
        frame_h: int,
        road_mask: np.ndarray | None,
        zones: list[SignZone],
    ) -> float | None:
        bbox = detection.bbox
        box_w = max(0.0, bbox.x2 - bbox.x1)
        box_h = max(0.0, bbox.y2 - bbox.y1)
        area = box_w * box_h

        if box_w <= 1.0 or box_h <= 1.0:
            return None

        height_ratio = box_h / float(frame_h)
        area_ratio = area / float(frame_w * frame_h)
        if height_ratio < self.min_height_ratio or area_ratio < self.min_area_ratio:
            return None

        road_support = self._mask_support(detection, road_mask)
        zone_support = max((self._mask_support(detection, zone.zone_mask) for zone in zones), default=0.0)

        detection.metadata["road_support"] = round(float(road_support), 4)
        detection.metadata["zone_support"] = round(float(zone_support), 4)

        if road_mask is not None and road_support < self.min_road_support and zone_support < self.min_zone_support:
            return None

        touches_edge = bbox.x1 <= self.edge_margin_px or bbox.x2 >= (frame_w - 1 - self.edge_margin_px)
        width_ratio = box_w / float(frame_w)
        if touches_edge and width_ratio <= self.edge_max_width_ratio and zone_support < self.min_zone_support:
            return None

        return (
            float(detection.confidence)
            + 0.35 * road_support
            + 0.45 * zone_support
            + 0.20 * min(height_ratio / 0.20, 1.0)
        )

    @staticmethod
    def _dilate_mask(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None or getattr(mask, "size", 0) == 0:
            return None
        normalized = np.where(mask > 0, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        return cv2.dilate(normalized, kernel, iterations=1)

    @staticmethod
    def _support_points(detection: Detection) -> list[tuple[float, float]]:
        bbox = detection.bbox
        return [
            (0.50 * (bbox.x1 + bbox.x2), bbox.y2),
            (bbox.x1 + 0.25 * (bbox.x2 - bbox.x1), bbox.y2 - 2),
            (bbox.x1 + 0.50 * (bbox.x2 - bbox.x1), bbox.y2 - 4),
            (bbox.x1 + 0.75 * (bbox.x2 - bbox.x1), bbox.y2 - 2),
            (0.50 * (bbox.x1 + bbox.x2), bbox.y1 + 0.78 * (bbox.y2 - bbox.y1)),
        ]

    def _mask_support(self, detection: Detection, mask: np.ndarray | None) -> float:
        if mask is None or getattr(mask, "size", 0) == 0:
            return 0.0

        h, w = mask.shape[:2]
        bbox = detection.bbox
        x1 = int(np.clip(np.floor(bbox.x1), 0, w - 1))
        x2 = int(np.clip(np.ceil(bbox.x2), 0, w - 1))
        y1 = int(np.clip(np.floor(bbox.y1 + 0.62 * (bbox.y2 - bbox.y1)), 0, h - 1))
        y2 = int(np.clip(np.ceil(bbox.y2), 0, h - 1))

        footprint_ratio = 0.0
        if x2 > x1 and y2 > y1:
            footprint = mask[y1:y2 + 1, x1:x2 + 1] > 0
            footprint_ratio = float(np.count_nonzero(footprint) / max(1, footprint.size))

        hits = 0
        for x, y in self._support_points(detection):
            px = int(round(float(x)))
            py = int(round(float(y)))
            if 0 <= px < w and 0 <= py < h and mask[py, px] > 0:
                hits += 1
        point_ratio = hits / 5.0

        return float(np.clip(max(footprint_ratio, point_ratio), 0.0, 1.0))

    def _suppress_occluded(self, scored: list[tuple[Detection, float]]) -> list[Detection]:
        suppressed: set[int] = set()

        for first_index, (first, first_score) in enumerate(scored):
            first_area = self._bbox_area(first)
            if first_area <= 0.0:
                suppressed.add(first_index)
                continue
            for second_index in range(first_index + 1, len(scored)):
                if first_index in suppressed or second_index in suppressed:
                    continue

                second, second_score = scored[second_index]
                second_area = self._bbox_area(second)
                if second_area <= 0.0:
                    suppressed.add(second_index)
                    continue

                overlap = bbox_intersection_area(first.bbox, second.bbox) / min(first_area, second_area)
                if overlap < self.occlusion_overlap_threshold:
                    continue

                if first_score >= second_score:
                    second.metadata["candidate_filter"] = "occluded"
                    suppressed.add(second_index)
                else:
                    first.metadata["candidate_filter"] = "occluded"
                    suppressed.add(first_index)

        return [detection for index, (detection, _) in enumerate(scored) if index not in suppressed]

    @staticmethod
    def _bbox_area(detection: Detection) -> float:
        bbox = detection.bbox
        return max(0.0, bbox.x2 - bbox.x1) * max(0.0, bbox.y2 - bbox.y1)
