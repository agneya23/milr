#!/usr/bin/env bash

set -Eeo pipefail
trap 'echo "ERROR: Script failed at line $LINENO"; exit 1' ERR

# ----------------------------
# Paths
# ----------------------------
CONDA_BASE="/ivi/zfs/s0/original_homes/ydu/miniconda3"
ENV_PATH="$CONDA_BASE/envs/milr_latentseek"
PROJECT_DIR="/ivi/zfs/s0/original_homes/ydu/PythonWorkSpace/agneya/milr"
GENEVAL_DIR="$PROJECT_DIR/src/geneval"
REWARDS_DIR="$PROJECT_DIR/src/rewards"
MMDET_DIR="$GENEVAL_DIR/mmdetection"

export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="8.9"

# ----------------------------
# Conda setup
# ----------------------------
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ -d "$ENV_PATH" ]; then
    conda env remove -y -p "$ENV_PATH"
fi

conda create -y -p "$ENV_PATH" python=3.10
conda activate "$ENV_PATH"

cd "$PROJECT_DIR"

# ----------------------------
# Compiler + CUDA toolkit
# Needed for packages like flash-attn.
# mmcv-full itself will use a prebuilt wheel.
# ----------------------------
conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11
conda install -y -c nvidia \
    cuda-nvcc=12.1.105 \
    cuda-cudart-dev=12.1 \
    cuda-cccl=12.1

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/include/cccl:$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/targets/x86_64-linux/include/cccl:${CPATH:-}"
export C_INCLUDE_PATH="$CPATH"
export CPLUS_INCLUDE_PATH="$CPATH"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"

echo "Compiler/CUDA check:"
"$CC" --version
"$CXX" --version
"$CUDACXX" --version

# ----------------------------
# Python build tooling
# ----------------------------
python -m pip install --upgrade "pip<25" "setuptools<70" wheel packaging ninja

# ----------------------------
# Pin NumPy early
# Important for torch 2.1 / old OpenMMLab stack.
# ----------------------------
python -m pip install "numpy==1.26.4"

# ----------------------------
# Install PyTorch cu121
#
# Use torch 2.1.0 because OpenMMLab has a matching prebuilt
# mmcv-full wheel for cu121/torch2.1.0.
# ----------------------------
python -m pip install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch
import numpy as np
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("numpy:", np.__version__)
assert torch.__version__.startswith("2.1.0"), torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert np.__version__.startswith("1.26"), np.__version__
PY

# ----------------------------
# Install requirements, but skip packages we handle manually
#
# Skip:
# - torch/torchvision/torchaudio: pinned above
# - flash_attn: installed separately after torch
# - groundingdino-py: dependency conflict with supervision>=0.22.0
# - mmcv/mmcv-full/mmdet: handled separately
# - numpy: pinned to 1.26.4
# ----------------------------
REQ_TMP="/tmp/milr_requirements_filtered.txt"

grep -vE '^[[:space:]]*(torch|torchvision|torchaudio|flash_attn|flash-attn|groundingdino-py|mmcv|mmcv-full|mmdet|numpy)([[:space:]]|==|>=|<=|>|<|$)' requirements.txt > "$REQ_TMP"

python -m pip install --no-build-isolation -r "$REQ_TMP"

# ----------------------------
# Fix NumPy/OpenCV after requirements
#
# requirements/supervision may pull:
# - numpy 2.x, which breaks torch 2.1
# - opencv-python, which needs libGL.so.1 on clusters
#
# Replace them with:
# - numpy 1.26.4
# - opencv-python-headless
# ----------------------------
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
python -m pip install --force-reinstall "numpy==1.26.4" "opencv-python-headless==4.8.1.78"

python - <<'PY'
import numpy as np
import cv2
print("numpy after fix:", np.__version__)
print("cv2 after fix:", cv2.__version__)
assert np.__version__.startswith("1.26"), np.__version__
PY

# ----------------------------
# Install flash-attn separately
# ----------------------------
python -m pip install --no-build-isolation flash_attn==2.7.2.post1

# Re-pin NumPy again in case flash-attn build deps touched it
python -m pip install --force-reinstall "numpy==1.26.4"

# ----------------------------
# Install groundingdino-py without forcing supervision==0.6.0
# ----------------------------
python -m pip install --no-build-isolation --no-deps groundingdino-py

# ----------------------------
# Install OpenMMLab stack
# ----------------------------
python -m pip install -U openmim

# Install mmcv deps manually because we will install mmcv-full with --no-deps
python -m pip install addict yapf tomli packaging Pillow pyyaml regex

# IMPORTANT:
# Use prebuilt mmcv-full wheel. Do NOT compile mmcv-full from source.
# Use --no-deps so pip does not pull opencv-python back in.
python -m pip install --no-cache-dir --only-binary=:all: --no-deps \
    "mmcv-full==1.7.2" \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

# Final NumPy/OpenCV cleanup before import checks
python -m pip uninstall -y opencv-python opencv-contrib-python || true
python -m pip install --force-reinstall "numpy==1.26.4" "opencv-python-headless==4.8.1.78"

python - <<'PY'
import numpy as np
import torch
import cv2
import mmcv

print("numpy:", np.__version__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cv2:", cv2.__version__)
print("mmcv:", mmcv.__version__)

assert np.__version__.startswith("1.26"), np.__version__
assert torch.__version__.startswith("2.1.0"), torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert mmcv.__version__ == "1.7.2", mmcv.__version__
PY

# ----------------------------
# Geneval object detector
# ----------------------------
cd "$GENEVAL_DIR"

if [ ! -f "./evaluation/download_models.sh" ]; then
    echo "ERROR: Geneval download script not found at $GENEVAL_DIR/evaluation/download_models.sh"
    exit 1
fi

./evaluation/download_models.sh "object_detector/"

# ----------------------------
# mmdetection 2.x
# ----------------------------
cd "$GENEVAL_DIR"

if [ ! -d "$MMDET_DIR" ]; then
    git clone https://github.com/open-mmlab/mmdetection.git "$MMDET_DIR"
else
    echo "mmdetection already exists: $MMDET_DIR"
fi

cd "$MMDET_DIR"
git fetch --all
git checkout 2.x

# Old mmdetection 2.x install
python setup.py develop

python - <<'PY'
import mmdet
print("mmdet:", mmdet.__version__)
PY

# ----------------------------
# Rewards object detector
# ----------------------------
if [ ! -d "$REWARDS_DIR" ]; then
    echo "ERROR: rewards directory not found at $REWARDS_DIR"
    exit 1
fi

cd "$REWARDS_DIR"

if [ ! -f "./evaluation/download_models.sh" ]; then
    echo "ERROR: rewards download script not found at $REWARDS_DIR/evaluation/download_models.sh"
    exit 1
fi

./evaluation/download_models.sh "object_detector/"

echo "Setup completed successfully."