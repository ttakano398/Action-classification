from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from action import BlockGCNInferencer, build_model_input_clip
from pose import Coco17Mapper, RTMOEstimator
from schemas import ActionResult, PoseObservation
from settings import resolve_runtime_paths
from tracking import TrackManager
from util import DebugVisualizer, JsonlWriter
from video_input import VideoSource


class RuntimePipeline:
    def __init__(self, config: dict[str, Any], root_dir: str | Path, device_override: str | None = None):
        self.config = resolve_runtime_paths(config, root_dir)
        self.pose_cfg = self.config["pose"]
        self.action_cfg = self.config["action"]
        self.output_cfg = self.config["output"]
        self.input_cfg = self.config["input"]

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
        self.visualizer = DebugVisualizer(self.output_cfg)
        self.json_writer = None
        if self.output_cfg.get("save_json", False):
            self.json_writer = JsonlWriter(self.output_cfg["json_path"])

    def run(self, source: str | int) -> None:
        source_reader = VideoSource(
            source=source,
            width=int(self.input_cfg.get("width", 0) or 0),
            height=int(self.input_cfg.get("height", 0) or 0),
        )
        last_time = time.perf_counter()

        try:
            for packet in source_reader.frames():
                now = time.perf_counter()
                fps = 1.0 / max(now - last_time, 1e-6)
                last_time = now

                tracked = self._process_frame(packet.frame, packet.timestamp_sec)
                canvas = self.visualizer.draw(packet.frame, tracked, fps=fps)

                if self.output_cfg.get("draw_overlay", True):
                    cv2.imshow(self.output_cfg.get("window_name", "RTMO Debug"), canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
        finally:
            source_reader.release()
            if self.json_writer:
                self.json_writer.close()
            cv2.destroyAllWindows()

    def _process_frame(
        self,
        frame,
        timestamp_sec: float,
    ) -> list[tuple[PoseObservation, ActionResult]]:
        detections = self.pose_estimator.infer(frame)
        mapped = [self.mapper.to_coco17(item) for item in detections]
        tracked = self.track_manager.update(mapped)

        tracked_results: list[tuple[PoseObservation, ActionResult]] = []
        for observation in tracked:
            clip = self.track_manager.get_clip(int(observation.track_id))
            action = self._predict_action(clip)
            tracked_results.append((observation, action))

        if self.json_writer:
            self.json_writer.write(timestamp_sec=timestamp_sec, tracked=tracked_results)

        return tracked_results

    def _predict_action(self, clip):
        if clip is None:
            return ActionResult(label=None, score=0.0, state="warmup")

        model_input = build_model_input_clip(
            clip=clip,
            clip_len=int(self.action_cfg["clip_len"]),
            conf_thr=float(self.pose_cfg.get("conf_thr", 0.3)),
        )
        return self.action_predictor.predict(model_input)
