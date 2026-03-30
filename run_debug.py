from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import RuntimePipeline
from settings import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTMO -> BlockGCN debug runtime")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--input", default=None, help="Video path or webcam index")
    parser.add_argument("--device", default=None, help="Override device, for example cuda:0 or cpu")
    parser.add_argument("--save-json", action="store_true", help="Force JSONL output on for this run")
    parser.add_argument("--save-video", action="store_true", help="Force overlay video output on for this run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    config = load_config(args.config)
    if args.save_json:
        config.setdefault("output", {})
        config["output"]["save_json"] = True
    if args.save_video:
        config.setdefault("output", {})
        config["output"]["save_video"] = True

    input_source = args.input
    if input_source is None:
        input_source = str(config.get("input", {}).get("source", 0))

    source = int(input_source) if str(input_source).isdigit() else input_source
    runtime = RuntimePipeline(config=config, root_dir=root_dir, device_override=args.device)
    runtime.run(source)


if __name__ == "__main__":
    main()
