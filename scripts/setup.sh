#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
DOWNLOAD_RTMO="${DOWNLOAD_RTMO:-0}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-1}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python binary not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${INSTALL_APT_DEPS}" == "1" ]] && command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    SUDO=""
  fi

  ${SUDO} apt-get update
  ${SUDO} apt-get install -y \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    python3-venv
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio
python -m pip install openmim
mim install "mmengine>=0.10.0" "mmcv>=2.1.0"
python -m pip install \
  "mmpose>=1.3.0,<1.4.0" \
  "mmaction2>=1.2.0,<1.3.0" \
  numpy \
  scipy \
  opencv-python \
  pyyaml

mkdir -p "${ROOT_DIR}/artifacts"
mkdir -p "${ROOT_DIR}/checkpoints"

if [[ "${DOWNLOAD_RTMO}" == "1" ]]; then
  mim download mmpose \
    --config rtmo-s_8xb32-600e_body7-640x640 \
    --dest "${ROOT_DIR}/checkpoints"
fi

cat <<EOF

Setup complete.

Activate the environment with:
  source "${VENV_DIR}/bin/activate"

Run the debug pipeline with:
  python run_debug.py --config config/default.yaml --input /path/to/video.mp4

EOF
