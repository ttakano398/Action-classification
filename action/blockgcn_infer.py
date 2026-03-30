from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from schemas import ActionResult


class BlockGCNInferencer:
    """Placeholder BlockGCN integration point.

    The initial implementation intentionally keeps the action backend optional.
    When no supported checkpoint is configured, the runtime returns a stable
    placeholder state instead of failing. This keeps the debug overlay useful
    while pose/tracking are being validated.
    """

    def __init__(self, action_cfg: dict[str, Any]):
        self.labels = list(action_cfg.get("labels", []))
        self.checkpoint = action_cfg.get("checkpoint")
        self.backend = action_cfg.get("backend", "pending")
        self.ready = False

        if self.checkpoint:
            checkpoint_path = Path(self.checkpoint)
            if checkpoint_path.exists():
                self.ready = False

    def predict(self, clip: np.ndarray | None) -> ActionResult:
        if clip is None:
            return ActionResult(label=None, score=0.0, state="warmup")
        if not self.ready:
            return ActionResult(label=None, score=0.0, state="no_action_model")
        return ActionResult(label=None, score=0.0, state="unimplemented")
