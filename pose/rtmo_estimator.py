from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from schemas import PoseObservation


def _resolve_model_asset(model_dir: Path, model_name: str, suffix: str) -> str:
    matches = sorted(model_dir.glob(f"{model_name}*{suffix}"))
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve {suffix} asset for '{model_name}' in {model_dir}. "
            "Set pose.config_file / pose.checkpoint explicitly or download the model."
        )
    return str(matches[0])


class RTMOEstimator:
    def __init__(self, pose_cfg: dict[str, Any], device_override: str | None = None):
        try:
            from mmpose.apis import inference_bottomup, init_model
        except ImportError as exc:
            raise RuntimeError(
                "MMPose is required to run RTMO. Run scripts/setup.sh on the target machine first."
            ) from exc

        self._inference_bottomup = inference_bottomup
        self.conf_thr = float(pose_cfg.get("conf_thr", 0.3))
        model_dir = Path(pose_cfg.get("model_dir", "checkpoints"))
        model_name = pose_cfg["model_name"]
        config_file = pose_cfg.get("config_file") or _resolve_model_asset(model_dir, model_name, ".py")
        checkpoint = pose_cfg.get("checkpoint") or _resolve_model_asset(model_dir, model_name, ".pth")
        device = device_override or pose_cfg.get("device", "cuda:0")
        self.model = init_model(config_file, checkpoint, device=device)

    def infer(self, frame: np.ndarray) -> list[PoseObservation]:
        data_sample = self._inference_bottomup(self.model, frame)
        pred_instances = getattr(data_sample, "pred_instances", None)
        if pred_instances is None:
            return []

        keypoints = np.asarray(getattr(pred_instances, "keypoints", []), dtype=np.float32)
        keypoint_scores = np.asarray(getattr(pred_instances, "keypoint_scores", []), dtype=np.float32)
        if keypoints.size == 0:
            return []

        if keypoint_scores.ndim == 2:
            keypoints = np.concatenate([keypoints, keypoint_scores[..., None]], axis=-1)

        bboxes = getattr(pred_instances, "bboxes", None)
        bbox_scores = getattr(pred_instances, "bbox_scores", None)

        outputs: list[PoseObservation] = []
        for idx in range(keypoints.shape[0]):
            person_keypoints = keypoints[idx]
            if person_keypoints.shape[-1] == 2:
                conf = np.ones((person_keypoints.shape[0], 1), dtype=np.float32)
                person_keypoints = np.concatenate([person_keypoints, conf], axis=-1)

            visible = person_keypoints[:, 2] >= self.conf_thr
            if not np.any(visible):
                continue

            if bboxes is not None and len(bboxes) > idx:
                bbox = np.asarray(bboxes[idx], dtype=np.float32)
            else:
                xy = person_keypoints[visible, :2]
                x1, y1 = np.min(xy, axis=0)
                x2, y2 = np.max(xy, axis=0)
                bbox = np.asarray([x1, y1, x2, y2], dtype=np.float32)

            if bbox_scores is not None and len(bbox_scores) > idx:
                det_score = float(bbox_scores[idx])
            else:
                det_score = float(np.mean(person_keypoints[visible, 2]))

            outputs.append(
                PoseObservation(
                    bbox=bbox,
                    keypoints=person_keypoints.astype(np.float32),
                    detection_score=det_score,
                )
            )
        return outputs
