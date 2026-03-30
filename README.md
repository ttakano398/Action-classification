# RTMO -> BlockGCN PoC

This repository contains the initial implementation scaffold for a debug-first
human action classification pipeline:

`video/webcam -> RTMO pose estimation -> Hungarian tracking -> COCO-17 mapping -> debug overlay`

The current scope focuses on:

- Ubuntu + CUDA environment setup
- RTMO-s (`rtmo-s_8xb32-600e_body7-640x640`) integration
- COCO-17 canonical preprocessing
- lightweight tracking with Hungarian matching
- debug visualization with skeleton overlay, `track_id`, and action placeholder

The full BlockGCN inference integration is intentionally left as the next step.
The runtime already exposes the interface where a BlockGCN predictor will plug
in later. Until then, the overlay can still validate pose, tracking, and state
flow with `warmup` / `no_action_model` states.

## Repository layout

```text
action/          Action preprocessing and predictor interface
config/          YAML configuration files
output/          JSON writer and visualization utilities
pipeline/        End-to-end runtime orchestration
pose/            RTMO wrapper and keypoint mapping
scripts/         Environment setup scripts
tracking/        Hungarian assigner and track manager
video_input/     Video / webcam source wrapper
run_debug.py     Main debug runtime entrypoint
spec.md          Project specification
```

## Target environment

- Ubuntu 22.04 LTS preferred
- NVIDIA GPU
- CUDA-capable PyTorch
- Python 3.10 or newer

## Setup

Run the setup script on the Ubuntu CUDA machine:

```bash
bash scripts/setup.sh
```

Optional environment variables:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
PYTHON_BIN=python3
VENV_DIR=.venv
DOWNLOAD_RTMO=1
```

If `DOWNLOAD_RTMO=1` is set, the script attempts to download the RTMO model
assets with:

```bash
mim download mmpose --config rtmo-s_8xb32-600e_body7-640x640 --dest checkpoints
```

If model download is skipped, place the RTMO config and checkpoint under
`checkpoints/`, or point the YAML config directly to custom paths.

## Configuration

The default runtime config lives at:

```text
config/default.yaml
```

Important fields:

- `pose.model_name`
- `pose.config_file`
- `pose.checkpoint`
- `tracking.match_method`
- `action.clip_len`
- `output.draw_overlay`

If `pose.config_file` and `pose.checkpoint` are left empty, the runtime tries
to resolve them from `checkpoints/` using the selected `pose.model_name`.

## Run debug overlay

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

Run on a video:

```bash
python run_debug.py --config config/default.yaml --input /path/to/video.mp4
```

Run on webcam:

```bash
python run_debug.py --config config/default.yaml --input 0
```

Useful flags:

```bash
python run_debug.py \
  --config config/default.yaml \
  --input /path/to/video.mp4 \
  --device cuda:0 \
  --save-json
```

## Current output behavior

The runtime overlays:

- bbox
- skeleton edges and joints
- `track_id`
- `action_label`
- `action_score`
- `state`
- FPS

Without a BlockGCN checkpoint wired in, each tracked person will remain in a
non-final action state such as `warmup` or `no_action_model`. This is expected
for the initial scope.

## Next implementation step

The next milestone is BlockGCN model integration:

- model loader
- clip-to-logit inference
- smoothing with real action scores
- label overlay backed by the classifier instead of the placeholder
