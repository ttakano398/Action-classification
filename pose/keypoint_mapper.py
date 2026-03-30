from __future__ import annotations

from typing import Iterable

import numpy as np

from schemas import PoseObservation

COCO17_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO17_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


class Coco17Mapper:
    def __init__(self, source_order: str | Iterable[str] = "coco17"):
        if source_order == "coco17":
            self.index_map = list(range(len(COCO17_KEYPOINTS)))
        else:
            source_names = list(source_order)
            source_index = {name: idx for idx, name in enumerate(source_names)}
            self.index_map = [source_index[name] for name in COCO17_KEYPOINTS]

    def to_coco17(self, observation: PoseObservation) -> PoseObservation:
        keypoints = observation.keypoints[self.index_map].astype(np.float32)
        return PoseObservation(
            bbox=observation.bbox.astype(np.float32),
            keypoints=keypoints,
            detection_score=float(observation.detection_score),
            track_id=observation.track_id,
        )

    @staticmethod
    def reference_center(keypoints: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        if left_hip[2] > 0 and right_hip[2] > 0:
            center = (left_hip[:2] + right_hip[:2]) / 2.0
            return center.astype(np.float32)
        x1, y1, x2, y2 = bbox
        return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    @staticmethod
    def normalize(observation: PoseObservation) -> PoseObservation:
        keypoints = observation.keypoints.copy().astype(np.float32)
        center = Coco17Mapper.reference_center(keypoints, observation.bbox)
        bbox_height = max(float(observation.bbox[3] - observation.bbox[1]), 1.0)
        keypoints[:, 0] = (keypoints[:, 0] - center[0]) / bbox_height
        keypoints[:, 1] = (keypoints[:, 1] - center[1]) / bbox_height
        return PoseObservation(
            bbox=observation.bbox.astype(np.float32),
            keypoints=keypoints,
            detection_score=float(observation.detection_score),
            track_id=observation.track_id,
        )
