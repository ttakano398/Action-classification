from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray
    index: int
    timestamp_sec: float


class VideoSource:
    def __init__(self, source: str | int, width: int | None = None, height: int | None = None):
        self.source = int(source) if str(source).isdigit() else str(source)
        if isinstance(self.source, str) and not str(self.source).isdigit():
            source_path = Path(self.source)
            if not source_path.exists():
                raise FileNotFoundError(f"Input source not found: {source_path}")

        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open input source: {self.source}")

        if width:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def fps(self) -> float:
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else 30.0

    def frames(self) -> Generator[FramePacket, None, None]:
        index = 0
        fps = self.fps()
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            timestamp_sec = index / fps
            yield FramePacket(frame=frame, index=index, timestamp_sec=timestamp_sec)
            index += 1

    def release(self) -> None:
        self.capture.release()
