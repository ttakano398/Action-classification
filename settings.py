from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def resolve_runtime_paths(config: dict[str, Any], root_dir: str | Path) -> dict[str, Any]:
    root = Path(root_dir)
    resolved = dict(config)

    output_cfg = dict(resolved.get("output", {}))
    if output_cfg.get("json_path"):
        output_cfg["json_path"] = str((root / output_cfg["json_path"]).resolve())
    if output_cfg.get("video_path"):
        output_cfg["video_path"] = str((root / output_cfg["video_path"]).resolve())
    resolved["output"] = output_cfg

    pose_cfg = dict(resolved.get("pose", {}))
    if pose_cfg.get("model_dir"):
        pose_cfg["model_dir"] = str((root / pose_cfg["model_dir"]).resolve())
    resolved["pose"] = pose_cfg

    action_cfg = dict(resolved.get("action", {}))
    for key in ("checkpoint", "repo_dir", "label_map_file"):
        if action_cfg.get(key):
            action_cfg[key] = str((root / action_cfg[key]).resolve())
    resolved["action"] = action_cfg

    return resolved
