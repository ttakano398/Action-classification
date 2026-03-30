from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from action.label_maps import get_label_map
from schemas import ActionResult


@contextmanager
def _prepend_sys_path(path: Path | None) -> Iterator[None]:
    inserted = False
    if path and str(path) not in sys.path:
        sys.path.insert(0, str(path))
        inserted = True
    try:
        yield
    finally:
        if inserted and str(path) in sys.path:
            sys.path.remove(str(path))


def _load_label_map(action_cfg: dict[str, Any]) -> list[str]:
    label_space = str(action_cfg.get("pretrained_label_space", "ntu120"))
    label_map_file = action_cfg.get("label_map_file")
    if label_map_file:
        path = Path(label_map_file)
        labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if labels:
            return labels
    return get_label_map(label_space)


def _normalize_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
    for key in ("model", "state_dict", "weights"):
        if key in raw_state and isinstance(raw_state[key], dict):
            raw_state = raw_state[key]
            break

    normalized = {}
    for key, value in raw_state.items():
        if not hasattr(value, "shape"):
            continue
        clean_key = key[7:] if key.startswith("module.") else key
        normalized[clean_key] = value
    return normalized


class BlockGCNInferencer:
    """Generic skeleton-GCN inferencer with an initial CTR-GCN backend.

    The class name is kept for backward compatibility with the existing
    pipeline import path, but the implementation now targets the initial
    CTR-GCN milestone.
    """

    def __init__(self, action_cfg: dict[str, Any]):
        self.action_cfg = dict(action_cfg)
        self.backend = str(self.action_cfg.get("model", self.action_cfg.get("backend", "pending"))).lower()
        self.repo_dir = Path(self.action_cfg.get("repo_dir", "third_party/CTR-GCN"))
        self.checkpoint = self.action_cfg.get("checkpoint")
        self.device = str(self.action_cfg.get("device", "cuda:0"))
        self.num_class = int(self.action_cfg.get("num_class", 120))
        self.num_point = int(self.action_cfg.get("num_joints", 25))
        self.num_person = int(self.action_cfg.get("num_person", 2))
        self.in_channels = int(self.action_cfg.get("in_channels", 3))
        self.graph = str(self.action_cfg.get("graph", "graph.ntu_rgb_d.Graph"))
        self.graph_args = dict(self.action_cfg.get("graph_args", {"labeling_mode": "spatial"}))
        self.model_module = str(self.action_cfg.get("model_module", "model.ctrgcn"))
        self.model_class = str(self.action_cfg.get("model_class", "Model"))
        self.labels = _load_label_map(self.action_cfg)

        self.ready = False
        self.not_ready_state = "no_action_model"
        self.model = None
        self.torch = None

        if self.backend not in {"ctrgcn", "pending"}:
            self.not_ready_state = f"unsupported_backend:{self.backend}"
            return
        if not self.checkpoint:
            self.not_ready_state = "no_action_model"
            return
        if not Path(self.checkpoint).exists():
            self.not_ready_state = "missing_action_checkpoint"
            return
        if not self.repo_dir.exists():
            self.not_ready_state = "missing_action_repo"
            return

        self._initialize_ctrgcn()

    def _initialize_ctrgcn(self) -> None:
        try:
            import torch
        except ImportError:
            self.not_ready_state = "missing_torch"
            return

        with _prepend_sys_path(self.repo_dir):
            try:
                module = importlib.import_module(self.model_module)
                model_cls = getattr(module, self.model_class)
            except Exception:
                self.not_ready_state = "missing_action_repo"
                return

            try:
                model = model_cls(
                    num_class=self.num_class,
                    num_point=self.num_point,
                    num_person=self.num_person,
                    graph=self.graph,
                    graph_args=self.graph_args,
                    in_channels=self.in_channels,
                )
                checkpoint = torch.load(self.checkpoint, map_location="cpu")
                state_dict = _normalize_state_dict(checkpoint)
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                self.not_ready_state = "action_model_init_failed"
                return

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"

        self.torch = torch
        self.model = model.to(self.device)
        self.model.eval()
        self.ready = True
        self.not_ready_state = "ready"

    def predict(self, clip: np.ndarray | None) -> ActionResult:
        if clip is None:
            return ActionResult(label=None, score=0.0, state="warmup")
        if not self.ready or self.model is None or self.torch is None:
            return ActionResult(label=None, score=0.0, state=self.not_ready_state)

        torch = self.torch
        input_tensor = torch.from_numpy(clip).float().unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            score, index = torch.max(probs, dim=1)

        label_idx = int(index.item())
        label = self._resolve_label(label_idx)
        return ActionResult(label=label, score=float(score.item()), state="raw")

    def _resolve_label(self, label_idx: int) -> str:
        if 0 <= label_idx < len(self.labels):
            return self.labels[label_idx]
        return f"class_{label_idx}"
