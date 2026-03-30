from __future__ import annotations

import numpy as np

from pose.keypoint_mapper import Coco17Mapper
from schemas import PoseObservation


def forward_fill_confident_keypoints(clip: np.ndarray, conf_thr: float) -> np.ndarray:
    filled = clip.copy().astype(np.float32)
    if filled.ndim != 3:
        raise ValueError(f"Expected clip shape [T, V, C], got {filled.shape}")

    for t in range(1, filled.shape[0]):
        current = filled[t]
        previous = filled[t - 1]
        missing = current[:, 2] < conf_thr
        current[missing, :2] = previous[missing, :2]
    return filled


def build_model_input_clip(
    clip: list[PoseObservation] | np.ndarray,
    clip_len: int,
    conf_thr: float,
) -> np.ndarray:
    if clip is None or len(clip) == 0:
        raise ValueError("Empty clip provided to build_model_input_clip")

    if isinstance(clip, np.ndarray):
        clip_array = clip.astype(np.float32)
    else:
        normalized_frames = [
            Coco17Mapper.normalize(observation).keypoints.astype(np.float32)
            for observation in clip
        ]
        clip_array = np.stack(normalized_frames, axis=0)

    if clip_array.shape[0] >= clip_len:
        clip_array = clip_array[-clip_len:]
    else:
        pad_count = clip_len - clip_array.shape[0]
        pad = np.repeat(clip_array[:1], pad_count, axis=0)
        clip_array = np.concatenate([pad, clip_array], axis=0)

    clip_array = forward_fill_confident_keypoints(clip_array, conf_thr=conf_thr)

    # [T, V, C] -> [C, T, V, M]
    clip_array = np.transpose(clip_array, (2, 0, 1))
    clip_array = clip_array[..., None]
    return clip_array.astype(np.float32)
