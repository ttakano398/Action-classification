from __future__ import annotations

import numpy as np

from pose.keypoint_mapper import COCO17_KEYPOINTS

NTU25_KEYPOINTS = [
    "spine_base",
    "spine_mid",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "spine_shoulder",
    "left_hand_tip",
    "left_thumb",
    "right_hand_tip",
    "right_thumb",
]

COCO17_INDEX = {name: idx for idx, name in enumerate(COCO17_KEYPOINTS)}


def _copy_joint(keypoints: np.ndarray, name: str, conf_scale: float = 1.0) -> np.ndarray:
    joint = keypoints[COCO17_INDEX[name]].copy().astype(np.float32)
    joint[2] *= float(conf_scale)
    return joint


def _midpoint(keypoints: np.ndarray, name_a: str, name_b: str) -> np.ndarray:
    joint_a = keypoints[COCO17_INDEX[name_a]].astype(np.float32)
    joint_b = keypoints[COCO17_INDEX[name_b]].astype(np.float32)
    midpoint = np.zeros(3, dtype=np.float32)
    midpoint[:2] = (joint_a[:2] + joint_b[:2]) / 2.0
    midpoint[2] = min(float(joint_a[2]), float(joint_b[2]))
    return midpoint


def _proxy_joint(keypoints: np.ndarray, name: str, conf_scale: float) -> np.ndarray:
    return _copy_joint(keypoints, name=name, conf_scale=conf_scale)


def _missing_proxy(keypoints: np.ndarray, name: str) -> np.ndarray:
    joint = _copy_joint(keypoints, name=name, conf_scale=1.0)
    joint[2] = 0.0
    return joint


def coco17_to_ntu25(keypoints: np.ndarray, proxy_conf_scale: float = 0.5) -> np.ndarray:
    if keypoints.shape[0] != len(COCO17_KEYPOINTS) or keypoints.shape[1] < 3:
        raise ValueError(f"Expected keypoints shape [17, 3+], got {keypoints.shape}")

    output = np.zeros((len(NTU25_KEYPOINTS), 3), dtype=np.float32)

    spine_base = _midpoint(keypoints, "left_hip", "right_hip")
    spine_shoulder = _midpoint(keypoints, "left_shoulder", "right_shoulder")
    spine_mid = np.zeros(3, dtype=np.float32)
    spine_mid[:2] = (spine_base[:2] + spine_shoulder[:2]) / 2.0
    spine_mid[2] = min(float(spine_base[2]), float(spine_shoulder[2]))

    output[0] = spine_base
    output[1] = spine_mid
    output[2] = spine_shoulder
    output[3] = _copy_joint(keypoints, "nose")
    output[4] = _copy_joint(keypoints, "left_shoulder")
    output[5] = _copy_joint(keypoints, "left_elbow")
    output[6] = _copy_joint(keypoints, "left_wrist")
    output[7] = _proxy_joint(keypoints, "left_wrist", conf_scale=proxy_conf_scale)
    output[8] = _copy_joint(keypoints, "right_shoulder")
    output[9] = _copy_joint(keypoints, "right_elbow")
    output[10] = _copy_joint(keypoints, "right_wrist")
    output[11] = _proxy_joint(keypoints, "right_wrist", conf_scale=proxy_conf_scale)
    output[12] = _copy_joint(keypoints, "left_hip")
    output[13] = _copy_joint(keypoints, "left_knee")
    output[14] = _copy_joint(keypoints, "left_ankle")
    output[15] = _proxy_joint(keypoints, "left_ankle", conf_scale=proxy_conf_scale)
    output[16] = _copy_joint(keypoints, "right_hip")
    output[17] = _copy_joint(keypoints, "right_knee")
    output[18] = _copy_joint(keypoints, "right_ankle")
    output[19] = _proxy_joint(keypoints, "right_ankle", conf_scale=proxy_conf_scale)
    output[20] = spine_shoulder
    output[21] = _missing_proxy(keypoints, "left_wrist")
    output[22] = _missing_proxy(keypoints, "left_wrist")
    output[23] = _missing_proxy(keypoints, "right_wrist")
    output[24] = _missing_proxy(keypoints, "right_wrist")
    return output
