#!/usr/bin/env bash
# Configure a fresh Nebius GPU box to run GVHMR. Idempotent; safe to re-run.
#
# Target: ubuntu22.04-cuda12 image on gpu-h200-sxm (Hopper, sm_90).
#
# Do NOT "upgrade" anything here. GVHMR pins a prebuilt pytorch3d wheel:
#     pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl   (py310 / cu121 / torch 2.3.0)
# Change the Python version, the torch version, or the architecture and that wheel stops
# applying, at which point pytorch3d builds from source and the setup goes from minutes to hours.
# Hopper (sm_90) is supported by CUDA 12.1; Blackwell (B200, RTX PRO 6000) is NOT, which is why
# this targets H200 despite it being more GPU than the job needs.
set -euo pipefail

GVHMR_SHA="${GVHMR_SHA:-main}"     # pin to a specific commit once a known-good one is confirmed
VENV="$HOME/venvs/gvhmr"
REPO="$HOME/GVHMR"

echo "==> system packages"
sudo apt-get update -qq
# git-lfs is not actually needed (neither GVHMR nor GMR uses LFS, and HF downloads over HTTP),
# but it costs ten seconds and removes a whole class of "why are my weights 133 bytes" confusion.
sudo apt-get install -y -qq build-essential git git-lfs python3.10-venv python3.10-dev ffmpeg unzip

echo "==> python venv (3.10 — required by the pytorch3d wheel)"
python3.10 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q -U pip wheel setuptools

echo "==> cython + numpy BEFORE requirements"
# cython_bbox compiles at install time and needs numpy already importable; installing it as a
# transitive dep of requirements.txt fails because the build env cannot see numpy yet.
pip install -q cython "numpy==1.23.5"

echo "==> clone GVHMR @ ${GVHMR_SHA}"
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/zju3dv/GVHMR "$REPO"
fi
cd "$REPO"
git fetch --all -q
git checkout -q "$GVHMR_SHA"
echo "    commit: $(git rev-parse HEAD)"

echo "==> requirements (verbatim — do not edit the torch/pytorch3d pins)"
pip install -q -r requirements.txt
pip install -q -e .

echo "==> checkpoints from the ungated HuggingFace mirror"
# The upstream README points at Google Drive, which is painful headless. camenduru/GVHMR carries
# the same files ungated, so no HF token is needed.
mkdir -p inputs/checkpoints outputs
pip install -q -U "huggingface_hub[cli]"
hf download camenduru/GVHMR \
    --include 'gvhmr/*' 'hmr2/*' 'vitpose/*' 'yolo/*' \
    --local-dir inputs/checkpoints

# DPVO is deliberately absent. It is the only component needing the CUDA *toolkit* rather than
# just the driver, it wants Eigen 3.4 + torch-scatter + numba + pypose, and `-s` (static camera)
# means it is never called. Our exo tripods never moved.

echo "==> body models (expected to have been scp'd already)"
BM="$REPO/inputs/checkpoints/body_models"
mkdir -p "$BM"
for f in smplx/SMPLX_NEUTRAL.npz smpl/SMPL_NEUTRAL.pkl; do
  if [ -f "$BM/$f" ]; then echo "    ok  $f"; else echo "    MISSING  $BM/$f"; fi
done

echo "==> verification"
python - <<'PY'
import torch
print(f"    torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    device: {torch.cuda.get_device_name(0)}  capability sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
import pytorch3d, numpy
print(f"    pytorch3d {pytorch3d.__version__}  numpy {numpy.__version__}")
PY

cat <<'EOF'

==> setup complete.

Smoke test on GVHMR's own bundled clip BEFORE uploading anything of ours:
    source ~/venvs/gvhmr/bin/activate && cd ~/GVHMR
    python tools/demo/demo.py --video docs/example_video/tennis.mp4 -s

Then our clips (-s = static camera, --f_mm from the tripod-distance calibration):
    python tools/demo/demo_folder.py -f inputs/clips -d outputs -s
EOF
