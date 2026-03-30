from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2

from action import BlockGCNInferencer, build_model_input_clip
from action import ActionSmoother
from pose import Coco17Mapper, RTMOEstimator
from schemas import ActionResult, PoseObservation
from settings import resolve_runtime_paths
from tracking import TrackManager
from util import DebugVisualizer, JsonlWriter
from video_input import VideoSource


def _build_stdout_logger(enabled: bool) -> logging.Logger:
    logger = logging.getLogger("action_det.runtime")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[runtime] %(message)s"))
        logger.addHandler(handler)

    logger.disabled = not enabled
    return logger


class RuntimePipeline:
    def __init__(self, config: dict[str, Any], root_dir: str | Path, device_override: str | None = None):
        self.config = resolve_runtime_paths(config, root_dir)
        self.pose_cfg = self.config["pose"]
        self.action_cfg = self.config["action"]
        self.output_cfg = self.config["output"]
        self.input_cfg = self.config["input"]
        self.logging_cfg = self.config.get("logging", {})

        self.pose_estimator = RTMOEstimator(self.pose_cfg, device_override=device_override)
        self.mapper = Coco17Mapper(self.pose_cfg.get("source_order", "coco17"))
        self.track_manager = TrackManager(
            clip_len=int(self.action_cfg["clip_len"]),
            max_missing_frames=int(self.config["tracking"]["max_missing_frames"]),
            iou_weight=float(self.config["tracking"]["iou_weight"]),
            center_dist_weight=float(self.config["tracking"]["center_dist_weight"]),
            keypoint_dist_weight=float(self.config["tracking"]["keypoint_dist_weight"]),
        )
        self.action_predictor = BlockGCNInferencer(self.action_cfg)
        self.action_smoother = ActionSmoother(
            score_thr=float(self.action_cfg.get("score_thr", 0.6)),
            confirm_count=int(self.action_cfg.get("smooth_confirm_count", 3)),
        )
        self.visualizer = DebugVisualizer(self.output_cfg)
        self.logger = _build_stdout_logger(bool(self.logging_cfg.get("enable_stdout", True)))
        self.log_interval = max(int(self.logging_cfg.get("interval", 15)), 1)
        self.log_empty_frames = bool(self.logging_cfg.get("log_empty_frames", False))
        self.json_writer = None
        self.video_writer = None
        if self.output_cfg.get("save_json", False):
            self.json_writer = JsonlWriter(self.output_cfg["json_path"])

    def run(self, source: str | int) -> None:
        source_reader = VideoSource(
            source=source,
            width=int(self.input_cfg.get("width", 0) or 0),
            height=int(self.input_cfg.get("height", 0) or 0),
        )
        source_fps = source_reader.fps()
        last_time = time.perf_counter()
        self.logger.info(
            "start source=%s width=%s height=%s device=%s pose_model=%s",
            source,
            self.input_cfg.get("width"),
            self.input_cfg.get("height"),
            self.pose_cfg.get("device"),
            self.pose_cfg.get("model_name"),
        )

        try:
            for packet in source_reader.frames():
                now = time.perf_counter()
                fps = 1.0 / max(now - last_time, 1e-6)
                last_time = now

                tracked, detection_count = self._process_frame(packet.frame, packet.timestamp_sec)
                track_ids = [int(observation.track_id) for observation, _ in tracked if observation.track_id is not None]
                self._log_frame(
                    frame_index=packet.index,
                    timestamp_sec=packet.timestamp_sec,
                    fps=fps,
                    detection_count=detection_count,
                    tracked=tracked,
                )
                canvas = self.visualizer.draw(
                    packet.frame,
                    tracked,
                    fps=fps,
                    runtime_info={
                        "source": source,
                        "frame_index": packet.index,
                        "timestamp_sec": packet.timestamp_sec,
                        "detection_count": detection_count,
                        "track_count": len(tracked),
                        "track_ids": track_ids,
                    },
                )
                if self.output_cfg.get("save_video", False):
                    if self.video_writer is None:
                        self.video_writer = self._create_video_writer(canvas, source_fps)
                    if self.video_writer is not None:
                        self.video_writer.write(canvas)

                if self.output_cfg.get("draw_overlay", True):
                    cv2.imshow(self.output_cfg.get("window_name", "RTMO Debug"), canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
        finally:
            source_reader.release()
            if self.json_writer:
                self.json_writer.close()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()

    def _process_frame(
        self,
        frame,
        timestamp_sec: float,
    ) -> tuple[list[tuple[PoseObservation, ActionResult]], int]:
        detections = self.pose_estimator.infer(frame)
        mapped = [self.mapper.to_coco17(item) for item in detections]
        tracked = self.track_manager.update(mapped)

        tracked_results: list[tuple[PoseObservation, ActionResult]] = []
        for observation in tracked:
            clip = self.track_manager.get_clip(int(observation.track_id))
            raw_action = self._predict_action(clip)
            action = self.action_smoother.update(int(observation.track_id), raw_action)
            tracked_results.append((observation, action))

        self.action_smoother.prune(self.track_manager.tracks.keys())

        if self.json_writer:
            self.json_writer.write(timestamp_sec=timestamp_sec, tracked=tracked_results)

        return tracked_results, len(detections)

    def _predict_action(self, clip):
        if clip is None:
            return ActionResult(label=None, score=0.0, state="warmup")

        model_input = build_model_input_clip(
            clip=clip,
            clip_len=int(self.action_cfg["clip_len"]),
            conf_thr=float(self.pose_cfg.get("conf_thr", 0.3)),
        )
        return self.action_predictor.predict(model_input)

    def _log_frame(
        self,
        frame_index: int,
        timestamp_sec: float,
        fps: float,
        detection_count: int,
        tracked: list[tuple[PoseObservation, ActionResult]],
    ) -> None:
        should_log = frame_index == 0 or (frame_index + 1) % self.log_interval == 0
        if not should_log and not (self.log_empty_frames and detection_count == 0):
            return

        track_ids = [int(observation.track_id) for observation, _ in tracked if observation.track_id is not None]
        states = [action.state for _, action in tracked]
        self.logger.info(
            "frame=%d time=%.2fs fps=%.1f detections=%d tracks=%d ids=%s states=%s",
            frame_index,
            timestamp_sec,
            fps,
            detection_count,
            len(tracked),
            track_ids,
            states,
        )

    def _create_video_writer(self, canvas, fps: float):
        video_path = self.output_cfg.get("video_path")
        if not video_path:
            return None

        path = Path(video_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = canvas.shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else float(self.input_cfg.get("fps", 30)),
            (width, height),
        )
        if not writer.isOpened():
            self.logger.info("failed to open video writer: %s", path)
            return None
        self.logger.info("save_video enabled: %s", path)
        return writer
