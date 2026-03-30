from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from schemas import PoseObservation
from tracking.assigner import AssignmentWeights, HungarianAssigner


@dataclass
class Track:
    track_id: int
    latest: PoseObservation
    missing_frames: int = 0
    history: deque[PoseObservation] = field(default_factory=lambda: deque(maxlen=32))

    def append(self, observation: PoseObservation) -> None:
        self.latest = observation
        self.missing_frames = 0
        self.history.append(observation)


class TrackManager:
    def __init__(
        self,
        clip_len: int,
        max_missing_frames: int,
        iou_weight: float,
        center_dist_weight: float,
        keypoint_dist_weight: float,
    ):
        self.clip_len = clip_len
        self.max_missing_frames = max_missing_frames
        self.assigner = HungarianAssigner(
            AssignmentWeights(
                iou_weight=iou_weight,
                center_dist_weight=center_dist_weight,
                keypoint_dist_weight=keypoint_dist_weight,
            )
        )
        self.tracks: dict[int, Track] = {}
        self._next_track_id = 1

    def update(self, detections: list[PoseObservation]) -> list[PoseObservation]:
        active_ids = list(self.tracks.keys())
        previous = {track_id: self.tracks[track_id].latest for track_id in active_ids}
        matches, unmatched_track_rows, unmatched_detections = self.assigner.match(active_ids, previous, detections)

        for track_id, det_idx in matches:
            obs = detections[det_idx]
            obs.track_id = track_id
            self.tracks[track_id].append(obs)

        for row in unmatched_track_rows:
            track_id = active_ids[row]
            track = self.tracks[track_id]
            track.missing_frames += 1
            if track.missing_frames > self.max_missing_frames:
                del self.tracks[track_id]

        for det_idx in unmatched_detections:
            obs = detections[det_idx]
            track_id = self._next_track_id
            self._next_track_id += 1
            obs.track_id = track_id
            track = Track(track_id=track_id, latest=obs, history=deque(maxlen=self.clip_len))
            track.append(obs)
            self.tracks[track_id] = track

        tracked = [obs for obs in detections if obs.track_id is not None]
        tracked.sort(key=lambda item: int(item.track_id or 0))
        return tracked

    def get_clip(self, track_id: int) -> list[PoseObservation] | None:
        track = self.tracks.get(track_id)
        if track is None or not track.history:
            return None
        return list(track.history)
