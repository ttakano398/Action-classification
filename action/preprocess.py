from __future__ import annotations

import numpy as np

from action.ntu_adapter import coco17_to_ntu25
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
    target_layout: str = "coco17",
    confidence_mode: str = "input_channel",
    proxy_conf_scale: float = 0.5,
    num_person: int = 1,
) -> np.ndarray:
    if clip is None or len(clip) == 0:
        raise ValueError("Empty clip provided to build_model_input_clip")

    if isinstance(clip, np.ndarray):
        clip_array = clip.astype(np.float32)
        if target_layout == "ntu25" and clip_array.shape[1] == 17:
            clip_array = np.stack(
                [coco17_to_ntu25(frame, proxy_conf_scale=proxy_conf_scale) for frame in clip_array],
                axis=0,
            )
    else:
        normalized_frames = [
            _convert_layout(
                Coco17Mapper.normalize(observation).keypoints.astype(np.float32),
                target_layout=target_layout,
                proxy_conf_scale=proxy_conf_scale,
            )
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
    clip_array = _apply_confidence_mode(clip_array, confidence_mode=confidence_mode)

    # [T, V, C] -> [C, T, V, M]
    clip_array = np.transpose(clip_array, (2, 0, 1))
    clip_array = clip_array[..., None]
    clip_array = _pad_person_dimension(clip_array, num_person=num_person)
    return clip_array.astype(np.float32)


def _convert_layout(keypoints: np.ndarray, target_layout: str, proxy_conf_scale: float) -> np.ndarray:
    if target_layout == "coco17":
        return keypoints
    if target_layout == "ntu25":
        return coco17_to_ntu25(keypoints, proxy_conf_scale=proxy_conf_scale)
    raise ValueError(f"Unsupported target_layout: {target_layout}")


def _apply_confidence_mode(clip_array: np.ndarray, confidence_mode: str) -> np.ndarray:
    if confidence_mode == "input_channel":
        return clip_array
    if confidence_mode == "zero":
        clip = clip_array.copy()
        clip[:, :, 2] = 0.0
        return clip
    raise ValueError(f"Unsupported confidence_mode: {confidence_mode}")


def _pad_person_dimension(clip_array: np.ndarray, num_person: int) -> np.ndarray:
    if num_person < 1:
        raise ValueError(f"num_person must be >= 1, got {num_person}")

    current = clip_array.shape[-1]
    if current == num_person:
        return clip_array
    if current > num_person:
        return clip_array[..., :num_person]

    padded = np.zeros((*clip_array.shape[:-1], num_person), dtype=clip_array.dtype)
    padded[..., :current] = clip_array
    return padded
