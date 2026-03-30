from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from schemas import PoseObservation


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def bbox_center(box: np.ndarray) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def mean_keypoint_distance(a: np.ndarray, b: np.ndarray) -> float:
    visible = (a[:, 2] > 0) & (b[:, 2] > 0)
    if not np.any(visible):
        return 1.0
    distances = np.linalg.norm(a[visible, :2] - b[visible, :2], axis=1)
    return float(np.mean(distances))


@dataclass
class AssignmentWeights:
    iou_weight: float
    center_dist_weight: float
    keypoint_dist_weight: float


class HungarianAssigner:
    def __init__(self, weights: AssignmentWeights):
        self.weights = weights

    def match(
        self,
        track_ids: Iterable[int],
        track_observations: dict[int, PoseObservation],
        detections: list[PoseObservation],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        track_ids = list(track_ids)
        if not track_ids or not detections:
            return [], list(range(len(track_ids))), list(range(len(detections)))

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise RuntimeError(
                "scipy is required for Hungarian matching. Install dependencies with scripts/setup.sh."
            ) from exc

        cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for row, track_id in enumerate(track_ids):
            previous = track_observations[track_id]
            prev_center = bbox_center(previous.bbox)
            prev_scale = max(previous.bbox[3] - previous.bbox[1], 1.0)
            for col, current in enumerate(detections):
                current_center = bbox_center(current.bbox)
                center_distance = np.linalg.norm(prev_center - current_center) / prev_scale
                keypoint_distance = mean_keypoint_distance(previous.keypoints, current.keypoints) / prev_scale
                overlap = bbox_iou(previous.bbox, current.bbox)
                cost = (
                    self.weights.iou_weight * (1.0 - overlap)
                    + self.weights.center_dist_weight * center_distance
                    + self.weights.keypoint_dist_weight * keypoint_distance
                )
                cost_matrix[row, col] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(range(len(track_ids)))
        unmatched_detections = set(range(len(detections)))
        for row, col in zip(row_ind.tolist(), col_ind.tolist()):
            matches.append((track_ids[row], col))
            unmatched_tracks.discard(row)
            unmatched_detections.discard(col)

        return matches, sorted(unmatched_tracks), sorted(unmatched_detections)
