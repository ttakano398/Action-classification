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


def _bbox_height(bbox: np.ndarray) -> float:
    return float(max(bbox[3] - bbox[1], 0.0))


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(float(ax1), float(bx1))
    inter_y1 = max(float(ay1), float(by1))
    inter_x2 = min(float(ax2), float(bx2))
    inter_y2 = min(float(ay2), float(by2))

    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(float(ax2 - ax1), 0.0) * max(float(ay2 - ay1), 0.0)
    area_b = max(float(bx2 - bx1), 0.0) * max(float(by2 - by1), 0.0)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class RTMOEstimator:
    def __init__(self, pose_cfg: dict[str, Any], device_override: str | None = None):
        try:
            from mmpose.apis import inference_bottomup, init_model
        except ImportError as exc:
            missing_module = getattr(exc, "name", None)
            if missing_module == "mmdet":
                detail = "The RTMO stack needs MMDetection (`mmdet`) in addition to MMPose."
            else:
                detail = f"Import failed while loading the RTMO stack: {exc}."
            raise RuntimeError(
                "The RTMO runtime stack is incomplete. "
                f"{detail} Run scripts/setup.sh on the target machine again."
            ) from exc

        self._inference_bottomup = inference_bottomup
        self.conf_thr = float(pose_cfg.get("conf_thr", 0.5))
        self.min_visible_joints = int(pose_cfg.get("min_visible_joints", 6))
        self.min_detection_score = float(pose_cfg.get("min_detection_score", 0.4))
        self.min_bbox_height = float(pose_cfg.get("min_bbox_height", 80.0))
        self.duplicate_iou_thr = float(pose_cfg.get("duplicate_iou_thr", 0.6))
        model_dir = Path(pose_cfg.get("model_dir", "checkpoints"))
        model_name = pose_cfg["model_name"]
        config_file = pose_cfg.get("config_file") or _resolve_model_asset(model_dir, model_name, ".py")
        checkpoint = pose_cfg.get("checkpoint") or _resolve_model_asset(model_dir, model_name, ".pth")
        device = device_override or pose_cfg.get("device", "cuda:0")
        self.model = init_model(config_file, checkpoint, device=device)

    def infer(self, frame: np.ndarray) -> list[PoseObservation]:
        results = self._inference_bottomup(self.model, frame)
        if isinstance(results, list):
            if not results:
                return []
            data_sample = results[0]
        else:
            data_sample = results

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
            visible_count = int(np.count_nonzero(visible))
            if visible_count < self.min_visible_joints:
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

            if det_score < self.min_detection_score:
                continue
            if _bbox_height(bbox) < self.min_bbox_height:
                continue

            outputs.append(
                PoseObservation(
                    bbox=bbox,
                    keypoints=person_keypoints.astype(np.float32),
                    detection_score=det_score,
                )
            )

        outputs.sort(key=lambda item: item.detection_score, reverse=True)
        deduped: list[PoseObservation] = []
        for observation in outputs:
            if any(_bbox_iou(observation.bbox, kept.bbox) >= self.duplicate_iou_thr for kept in deduped):
                continue
            deduped.append(observation)

        return deduped
