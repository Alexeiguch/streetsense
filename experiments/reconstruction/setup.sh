#!/usr/bin/env bash
set -euo pipefail
RECON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RECON_CACHE="$RECON_ROOT/.cache/reconstruction"
RECON_ENV="$RECON_ROOT/.venv-3d"
RECON_PYTHON="${1:-$RECON_ROOT/.venv/bin/python}"
mkdir -p "$RECON_CACHE/torch/hub"
"$RECON_PYTHON" -m venv --system-site-packages "$RECON_ENV"
# Reuse the installed matching torch/torchvision pair; never change the production env.
"$RECON_ENV/bin/python" -c 'import torch, torchvision, cv2, numpy, transformers'
if [[ ! -d "$RECON_CACHE/map-anything/.git" ]]; then
  git clone https://github.com/facebookresearch/map-anything.git "$RECON_CACHE/map-anything"
fi
git -C "$RECON_CACHE/map-anything" checkout --detach 3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9
if [[ ! -d "$RECON_CACHE/torch/hub/facebookresearch_dinov2_main/.git" ]]; then
  git clone https://github.com/facebookresearch/dinov2.git "$RECON_CACHE/torch/hub/facebookresearch_dinov2_main"
fi
# UniCeption looks for this Torch Hub directory. Populate it with a pinned official checkout.
git -C "$RECON_CACHE/torch/hub/facebookresearch_dinov2_main" checkout --detach 7764ea0f912e53c92e82eb78a2a1631e92725fc8
"$RECON_ENV/bin/python" -m pip install --no-deps -e "$RECON_CACHE/map-anything" uniception==0.1.7
"$RECON_ENV/bin/python" -m pip install -r "$RECON_ROOT/experiments/reconstruction/requirements-inference.txt"
"$RECON_ENV/bin/python" -c 'from mapanything.models import MapAnything; print("3D inference environment ready")'
