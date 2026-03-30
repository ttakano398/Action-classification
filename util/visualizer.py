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
        runtime_info: dict | None = None,
    ) -> np.ndarray:
        canvas = frame.copy()
        for observation, action in tracked:
            self._draw_bbox(canvas, observation)
            self._draw_skeleton(canvas, observation)
            self._draw_label(canvas, observation, action)

        self._draw_runtime_hud(canvas, fps=fps, runtime_info=runtime_info)
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
        label = " | ".join(tokens)
        origin = (x1, max(20, y1 - 10))
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2,
        )
        top_left = (origin[0] - 4, origin[1] - text_height - 6)
        bottom_right = (origin[0] + text_width + 4, origin[1] + baseline + 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)
        cv2.putText(
            frame,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_runtime_hud(
        self,
        frame: np.ndarray,
        fps: float | None = None,
        runtime_info: dict | None = None,
    ) -> None:
        lines: list[str] = []

        if self.output_cfg.get("show_runtime_status", True) and runtime_info:
            source = runtime_info.get("source")
            frame_index = runtime_info.get("frame_index")
            timestamp_sec = runtime_info.get("timestamp_sec")
            detection_count = runtime_info.get("detection_count")
            track_count = runtime_info.get("track_count")
            track_ids = runtime_info.get("track_ids")
            lines.append(f"src={source} frame={frame_index} t={timestamp_sec:.2f}s")
            lines.append(f"det={detection_count} trk={track_count} ids={track_ids}")

        if self.output_cfg.get("show_fps", True) and fps is not None:
            lines.append(f"fps={fps:.1f}")

        if not lines:
            return

        x = 20
        y = 30
        line_height = 24
        max_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for line in lines)
        box_height = 12 + line_height * len(lines)
        cv2.rectangle(frame, (x - 10, y - 22), (x + max_width + 10, y - 22 + box_height), (0, 0, 0), -1)

        for idx, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x, y + idx * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255) if line.startswith("fps=") else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
