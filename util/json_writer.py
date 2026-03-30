from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from schemas import ActionResult, PoseObservation


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def write(
        self,
        timestamp_sec: float,
        tracked: Iterable[tuple[PoseObservation, ActionResult]],
    ) -> None:
        persons = []
        for observation, action in tracked:
            persons.append(
                {
                    "track_id": observation.track_id,
                    "bbox": observation.bbox.tolist(),
                    "keypoints": observation.keypoints.tolist(),
                    "action_label": action.label,
                    "action_score": action.score,
                    "state": action.state,
                }
            )
        record = {"timestamp": timestamp_sec, "persons": persons}
        self.handle.write(json.dumps(record) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()
