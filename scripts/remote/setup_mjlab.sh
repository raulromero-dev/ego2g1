#!/usr/bin/env bash
# Configure a Nebius GPU box to train G1 motion-tracking policies with mjlab.
#
# Target: ubuntu22.04-cuda12 image, NVIDIA GPU (mjlab trains on CUDA only; macOS is eval-only).
#
# Unlike the GVHMR box this one is not version-fragile -- mjlab installs through `uv` and pins its
# own dependencies. The fiddly part is motion preprocessing, documented below.
set -euo pipefail

REPO="$HOME/mjlab"
MOTIONS_IN="$HOME/motions_csv"     # LAFAN1-format CSVs, rsync'd from the laptop
MOTIONS_OUT="$HOME/motions_npz"    # converted, 50 Hz, ready for training

echo "==> system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential git curl ffmpeg

echo "==> uv (mjlab's expected installer)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "==> clone mjlab"
[ -d "$REPO/.git" ] || git clone https://github.com/mujocolab/mjlab "$REPO"
cd "$REPO"
uv sync

echo "==> patch csv_to_npz to stay local"
# csv_to_npz.py writes /tmp/motion.npz and then unconditionally uploads it to a Weights & Biases
# registry called "motions" -- there is no flag to skip it. We do not need it: training reads a
# local file via the env config's `motion_file`, and `registry_name` in the tracking runner is
# only used for `wandb.run.use_artifact()`, i.e. lineage tracking, not loading. Patching the
# upload out avoids requiring an account and avoids creating one W&B run per motion file
# (we have ~110 of them).
python3 - <<'PY'
from pathlib import Path
p = Path.home() / "mjlab/src/mjlab/scripts/csv_to_npz.py"
s = p.read_text()
marker = 'print("Uploading to Weights & Biases...")'
if marker in s and "EGO2G1_NO_WANDB" not in s:
    head, _sep, tail = s.partition(marker)
    indent = " " * (len(head.split("\n")[-1]))
    s = head + (
        "# EGO2G1_NO_WANDB: upload disabled; /tmp/motion.npz above is the artifact we keep.\n"
        + indent + "import os, shutil\n"
        + indent + "_dst = os.environ.get('MJLAB_NPZ_OUT')\n"
        + indent + "if _dst:\n"
        + indent + "  shutil.copy('/tmp/motion.npz', _dst)\n"
        + indent + "  print(f'[ego2g1] wrote {_dst}')\n"
        + indent + "return\n"
        + indent + 'print("Uploading to Weights & Biases...")' + tail
    )
    p.write_text(s)
    print("  patched")
else:
    print("  already patched or marker missing -- inspect manually")
PY

echo "==> verify"
uv run python - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device {torch.cuda.get_device_name(0)}  sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
import mujoco
print(f"  mujoco {mujoco.__version__}")
PY
uv run python -m mjlab.scripts.list_envs 2>/dev/null | grep -i tracking || \
  echo "  (list_envs did not print tracking tasks -- check task registration)"

mkdir -p "$MOTIONS_IN" "$MOTIONS_OUT"

cat <<'EOF'

==> setup complete.

Convert motions (one CSV at a time; MJLAB_NPZ_OUT tells the patch where to put the result):
    cd ~/mjlab
    for f in ~/motions_csv/*.csv; do
      n=$(basename "$f" .csv)
      MJLAB_NPZ_OUT=~/motions_npz/$n.npz uv run python -m mjlab.scripts.csv_to_npz \
        --input-file "$f" --output-name "$n" --input-fps 30 --output-fps 50
    done

Train (task Mjlab-Tracking-Flat-Unitree-G1; motion_file points at a local npz):
    uv run python -m mjlab.scripts.train --task Mjlab-Tracking-Flat-Unitree-G1 --help
EOF
