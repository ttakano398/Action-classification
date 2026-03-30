from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PoseObservation:
    bbox: np.ndarray
    keypoints: np.ndarray
    detection_score: float
    track_id: Optional[int] = None


@dataclass
class ActionResult:
    label: Optional[str]
    score: float
    state: str
