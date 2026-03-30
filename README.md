# RTMO -> CTR-GCN PoC

This repository contains the initial implementation scaffold for a debug-first
human action classification pipeline:

`video/webcam -> RTMO pose estimation -> Hungarian tracking -> COCO-17 mapping -> pseudo NTU-25 adapter -> debug overlay`

The current scope focuses on:

- Ubuntu + CUDA environment setup
- RTMO-s (`rtmo-s_8xb32-600e_body7-640x640`) integration
- COCO-17 canonical preprocessing
- `COCO-17 -> pseudo NTU-25` action-input adaptation
- lightweight tracking with Hungarian matching
- debug visualization with skeleton overlay, `track_id`, and action placeholder

The initial action backend target is `CTR-GCN`, with `HD-GCN` as a secondary
fallback and `BlockGCN` as the longer-term target. The runtime already exposes
the interface where a pretrained GCN predictor will plug in later. Until then,
the overlay can still validate pose, tracking, adapter, and state flow with
`warmup` / `no_action_model` states.

## Repository layout

```text
action/            Action preprocessing and predictor interface
config/            YAML configuration files
pipeline/          End-to-end runtime orchestration
pose/              RTMO wrapper and keypoint mapping
scripts/           Environment setup scripts
tracking/          Hungarian assigner and track manager
util/              Visualization and JSON writer utilities
video_input/       Video / webcam source wrapper
output/            Generated overlays and JSONL outputs
run_debug.py       Main debug runtime entrypoint
spec.md            Project specification
```

## Target environment

- Ubuntu 22.04 LTS preferred
- NVIDIA GPU
- CUDA-capable PyTorch
- Python 3.10 preferred
- Python 3.11 acceptable
- Python 3.13 is not supported by the current setup script

## Setup

### Prerequisites

Install these system packages on the Ubuntu CUDA machine before running the
project setup:

```bash
sudo apt-get update
sudo apt-get install -y git curl ffmpeg libgl1 libglib2.0-0 python3-venv
```

If you do not want `setup.sh` to invoke `sudo`, keep the default behavior and
install the packages above yourself.

### Python setup

Run the setup script on the Ubuntu CUDA machine:

```bash
bash scripts/setup.sh
```

Optional environment variables:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
PYTHON_BIN=python3.10
VENV_DIR=.venv
DOWNLOAD_RTMO=1
TORCH_VERSION=2.1.0
TORCHVISION_VERSION=0.16.0
MMENGINE_VERSION=0.10.7
MMCV_VERSION=2.1.0
MMCV_WHEEL_URL=https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
MMDET_VERSION=3.2.0
NUMPY_VERSION=1.26.4
INSTALL_APT_DEPS=1
INSTALL_MMACTION2=1
```

The setup intentionally installs `torch` and `torchvision` only. This project
does not use audio, so `torchaudio` is omitted to avoid unnecessary Python
wheel compatibility issues.

The default stack is pinned to `torch==2.1.0`, `torchvision==0.16.0`,
`mmengine==0.10.7`, `mmcv==2.1.0`, `mmdet==3.2.0`, and `numpy==1.26.4`. This
keeps the RTMO path on an older but more internally consistent OpenMMLab stack:
PyTorch documents CUDA 12.1 wheels for Torch 2.1.0, OpenMMLab provides a
prebuilt `mmcv==2.1.0` wheel for `cu121/torch2.1.0`, and MMPose 1.3.x expects
an MMDetection 3.x dependency for RTMO. This avoids the source-build path of
`mmcv`, the missing-`mmdet` import failure in `RTMOHead`, and NumPy ABI
mismatches in `xtcocotools`.

For the current PoC, `mmpose` is installed with `--no-deps` after the required
runtime libraries are installed manually. This intentionally skips `chumpy`,
which is listed in MMPose runtime requirements but is not needed by the current
RTMO debug pipeline and is a common source of installation failures on fresh
Ubuntu environments. This is an implementation choice for this repository,
based on the fact that the current code path only uses `mmpose.apis` for 2D
bottom-up inference, while RTMO still requires `mmdet` and therefore installs
that dependency explicitly.

`xtcocotools` is installed with `--no-build-isolation` after pinning
`numpy==1.26.4` so that its compiled extension matches the active NumPy ABI.
This avoids import-time errors such as `numpy.dtype size changed`.

`mmaction2` is optional for now because the current repository still uses a
placeholder BlockGCN backend. If you want to prepare the future action-model
environment in advance, set `INSTALL_MMACTION2=1`.

If `DOWNLOAD_RTMO=1` is set, the script attempts to download the RTMO model
assets with:

```bash
mim download mmpose --config rtmo-s_8xb32-600e_body7-640x640 --dest checkpoints
```

If model download is skipped, place the RTMO config and checkpoint under
`checkpoints/`, or point the YAML config directly to custom paths.

## Machine checklist

To keep the setup as generic as possible, it helps to know these details on the
target Ubuntu machine:

- `uname -m`
- `lsb_release -a`
- `python3.10 --version`
- `nvidia-smi`
- available GPU VRAM
- whether `sudo` is available
- whether GUI display is available for `cv2.imshow`
- webcam path such as `/dev/video0`

Useful collection commands:

```bash
uname -m
lsb_release -a
python3.10 --version
nvidia-smi
df -h
ls -l /dev/video0
```

For this PoC, the most important practical constraints are:

- x86_64 Ubuntu
- NVIDIA driver that matches CUDA 12.1 capable PyTorch
- at least one accessible camera device if you want live validation
- GUI session for overlay windows, or else save output instead of showing it

VRAM requirements are modest for `RTMO-s` inference, but more VRAM gives more
headroom once BlockGCN training or larger experiments begin.

## Validated environment

The setup is intended to stay generic, but it has been checked against this
Ubuntu CUDA environment:

- `x86_64`
- Ubuntu 22.04.5 LTS
- Python 3.10.12
- NVIDIA RTX A4500 20GB
- NVIDIA driver 580.126.09
- `/dev/video0` available
- GUI session available for `cv2.imshow`

This is a validation reference, not a hard requirement. The documented setup
should still work on similar Ubuntu 22.04 + NVIDIA GPU environments.

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
- `action.repo_dir`
- `action.checkpoint`
- `action.target_layout`
- `action.proxy_conf_scale`
- `output.draw_overlay`

If `pose.config_file` and `pose.checkpoint` are left empty, the runtime tries
to resolve them from `checkpoints/` using the selected `pose.model_name`.

For the current `CTR-GCN` path, `action.target_layout: ntu25` means the runtime
keeps pose and tracking in `COCO-17`, then converts each normalized clip to a
`pseudo NTU-25` layout right before action inference. Missing joints are
handled conservatively:

- direct joints keep their original confidence
- midpoint joints such as `spine_base` use the lower confidence of the parents
- proxy joints such as `left_hand` and `left_foot` reuse nearby coordinates
  with down-weighted confidence via `action.proxy_conf_scale`
- joints with no meaningful proxy such as `thumb` and `hand_tip` are filled
  with proxy coordinates but `confidence=0`

## Recommended CTR-GCN checkpoint

The recommended starting point for future `CTR-GCN` integration is:

- official `CTR-GCN`
- `joint` modality
- `NTU RGB+D 120` `cross-subject` pretrained checkpoint

This is the default recommendation because the official repository explicitly
supports NTU120 and provides pretrained models for NTU RGB+D 60 and 120 cross
subject, while also noting a strong joint-only baseline for NTU120. If weight
availability or debugging simplicity is a concern, the next fallback is a
joint-only `NTU RGB+D 60 cross-subject` checkpoint.

Place the official `CTR-GCN` repository at:

```text
third_party/CTR-GCN
```

Place the initial checkpoint at:

```text
checkpoints/ctrgcn/ntu120_xsub_joint.pt
```

These match the defaults in `config/default.yaml`. If you prefer another
layout, update:

- `action.repo_dir`
- `action.checkpoint`

The initial wrapper assumes the official repo layout and imports:

- `model.ctrgcn`
- `graph.ntu_rgb_d.Graph`

## Run debug overlay

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

Use the source configured in `config/default.yaml`:

```bash
python run_debug.py --config config/default.yaml
```

Run on a video:

```bash
python run_debug.py --config config/default.yaml --input /path/to/video.mp4
```

Run on webcam:

```bash
python run_debug.py --config config/default.yaml --input /dev/video0
```

Useful flags:

```bash
python run_debug.py \
  --config config/default.yaml \
  --input /path/to/video.mp4 \
  --device cuda:0 \
  --save-video \
  --save-json
```

## Tuning tips for skeleton detection

When the skeleton overlay is unstable, missing, or overly noisy, start by
adjusting the pose-related parameters in `config/default.yaml`.

If the skeleton is hard to detect or disappears too easily:

- lower `pose.conf_thr`
  - example: `0.5 -> 0.4`
- lower `pose.min_visible_joints`
  - example: `6 -> 5`
- lower `pose.min_detection_score`
  - example: `0.4 -> 0.3`
- lower `pose.min_bbox_height` if the person appears smaller in frame
  - example: `80 -> 60`

If false positives are frequent or duplicate boxes appear on the same person:

- raise `pose.conf_thr`
  - example: `0.5 -> 0.6`
- raise `pose.min_visible_joints`
  - example: `6 -> 8`
- raise `pose.min_detection_score`
  - example: `0.4 -> 0.5`
- raise `pose.min_bbox_height` to reject tiny detections
  - example: `80 -> 120`
- lower `pose.duplicate_iou_thr` to merge overlapping duplicate detections more aggressively
  - example: `0.6 -> 0.5`

Tracking-related cleanup can also help if IDs flicker or stale tracks remain:

- lower `tracking.max_missing_frames`
  - example: `8 -> 3`

Practical guidance:

- if nothing is detected, loosen thresholds first
- if too many noisy boxes appear, tighten pose thresholds first
- if the same person gets multiple overlapping boxes, tune `duplicate_iou_thr`
- if IDs remain after a person leaves, tune `max_missing_frames`

## Current output behavior

The runtime overlays:

- bbox
- skeleton edges and joints
- `track_id`
- `action_label`
- `action_score`
- `state`
- FPS

Without a valid `CTR-GCN` repo and checkpoint wired in, each tracked person
will remain in a non-final action state such as `warmup`,
`missing_action_checkpoint`, `missing_action_repo`, or `no_action_model`. This
is expected until the backend assets are placed in the configured paths. The
runtime already includes state management for future classifier outputs, so
once the real action backend is connected it can expose `stable`,
`transition`, and `uncertain` states without further pipeline changes.

## Next implementation step

The next milestone after the initial `CTR-GCN` wiring is:

- official checkpoint validation on Ubuntu
- stride-aware inference scheduling
- `NTU label` overlay backed by the classifier instead of the placeholder
- later comparison with `HD-GCN`, then migration path to `BlockGCN`
