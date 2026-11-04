#!/usr/bin/env bash
set -euo pipefail
RECON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME="$RECON_ROOT/.cache/reconstruction/hf"
export TORCH_HOME="$RECON_ROOT/.cache/reconstruction/torch"
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$RECON_ROOT"
exec "$RECON_ROOT/.venv-3d/bin/python" -m experiments.reconstruction "$@"
