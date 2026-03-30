#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.1.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.16.0}"
MMENGINE_VERSION="${MMENGINE_VERSION:-0.10.7}"
MMCV_VERSION="${MMCV_VERSION:-2.1.0}"
MMCV_WHEEL_URL="${MMCV_WHEEL_URL:-https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html}"
MMDET_VERSION="${MMDET_VERSION:-3.2.0}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
DOWNLOAD_RTMO="${DOWNLOAD_RTMO:-0}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-0}"
INSTALL_MMACTION2="${INSTALL_MMACTION2:-0}"

if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3.10 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python binary not found: ${PYTHON_BIN}" >&2
  exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "${PYTHON_VERSION}" in
  3.10|3.11)
    ;;
  *)
    echo "Unsupported Python version: ${PYTHON_VERSION}" >&2
    echo "Use PYTHON_BIN=python3.10 (preferred) or PYTHON_BIN=python3.11." >&2
    exit 1
    ;;
esac

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  EXISTING_VENV_VERSION="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${EXISTING_VENV_VERSION}" != "${PYTHON_VERSION}" ]]; then
    echo "Existing virtualenv uses Python ${EXISTING_VENV_VERSION}, but ${PYTHON_BIN} is Python ${PYTHON_VERSION}." >&2
    echo "Remove ${VENV_DIR} and rerun the setup, or choose a matching VENV_DIR." >&2
    exit 1
  fi
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
python -m pip install \
  --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"
python -m pip install openmim
python -m pip install "mmengine==${MMENGINE_VERSION}"
python -m pip install \
  "mmcv==${MMCV_VERSION}" \
  -f "${MMCV_WHEEL_URL}"
python -m pip install \
  "numpy==${NUMPY_VERSION}" \
  cython \
  json_tricks \
  matplotlib \
  munkres \
  opencv-python \
  pillow \
  pyyaml
python -m pip install \
  scipy
python -m pip install \
  --no-build-isolation \
  "xtcocotools>=1.12"
python -m pip install "mmdet==${MMDET_VERSION}"
python -m pip install --no-deps "mmpose>=1.3.0,<1.4.0"

if [[ "${INSTALL_MMACTION2}" == "1" ]]; then
  python -m pip install "mmaction2>=1.2.0,<1.3.0"
fi

mkdir -p "${ROOT_DIR}/output"
mkdir -p "${ROOT_DIR}/checkpoints"

if [[ "${DOWNLOAD_RTMO}" == "1" ]]; then
  mim download mmpose \
    --config rtmo-s_8xb32-600e_body7-640x640 \
    --dest "${ROOT_DIR}/checkpoints"
fi

cat <<EOF

Setup complete.

Selected Python:
  ${PYTHON_BIN} (${PYTHON_VERSION})

Pinned runtime stack:
  torch==${TORCH_VERSION}
  torchvision==${TORCHVISION_VERSION}
  mmengine==${MMENGINE_VERSION}
  mmcv==${MMCV_VERSION}
  mmdet==${MMDET_VERSION}
  numpy==${NUMPY_VERSION}
  mmpose>=1.3.0,<1.4.0 (installed without chumpy)

Activate the environment with:
  source "${VENV_DIR}/bin/activate"

Run the debug pipeline with:
  python run_debug.py --config config/default.yaml --input /path/to/video.mp4

EOF
