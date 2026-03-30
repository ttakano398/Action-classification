from __future__ import annotations

import cv2
import numpy as np

from pose.keypoint_mapper import COCO17_EDGES
from schemas import ActionResult, PoseObservation


class DebugVisualizer:
    def __init__(self, output_cfg: dict):
        self.output_cfg = output_cfg

    def draw(
        self,
        frame: np.ndarray,
        tracked: list[tuple[PoseObservation, ActionResult]],
        fps: float | None = None,
    ) -> np.ndarray:
        canvas = frame.copy()
        for observation, action in tracked:
            self._draw_bbox(canvas, observation)
            self._draw_skeleton(canvas, observation)
            self._draw_label(canvas, observation, action)

        if self.output_cfg.get("show_fps", True) and fps is not None:
            cv2.putText(
                canvas,
                f"FPS: {fps:.1f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return canvas

    @staticmethod
    def _draw_bbox(frame: np.ndarray, observation: PoseObservation) -> None:
        x1, y1, x2, y2 = observation.bbox.astype(int).tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    @staticmethod
    def _draw_skeleton(frame: np.ndarray, observation: PoseObservation) -> None:
        keypoints = observation.keypoints
        for start, end in COCO17_EDGES:
            if keypoints[start, 2] <= 0 or keypoints[end, 2] <= 0:
                continue
            pt1 = tuple(np.round(keypoints[start, :2]).astype(int).tolist())
            pt2 = tuple(np.round(keypoints[end, :2]).astype(int).tolist())
            cv2.line(frame, pt1, pt2, (255, 128, 0), 2)

        for point in keypoints:
            if point[2] <= 0:
                continue
            x, y = np.round(point[:2]).astype(int).tolist()
            cv2.circle(frame, (x, y), 3, (0, 128, 255), -1)

    def _draw_label(self, frame: np.ndarray, observation: PoseObservation, action: ActionResult) -> None:
        x1, y1, _, _ = observation.bbox.astype(int).tolist()
        tokens = []
        if self.output_cfg.get("show_track_id", True):
            tokens.append(f"id={observation.track_id}")
        if self.output_cfg.get("show_action_label", True):
            tokens.append(f"action={action.label or '-'}")
        if self.output_cfg.get("show_action_score", True):
            tokens.append(f"score={action.score:.2f}")
        tokens.append(f"state={action.state}")
        cv2.putText(
            frame,
            " | ".join(tokens),
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
