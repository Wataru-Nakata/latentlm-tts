#!/usr/bin/env bash
# Convenience launcher for the no-container (plain-PyTorch) training path.
#
#   ./scripts/launch.sh [NGPU] [CONFIG] [-- extra train_lite args]
#
# Examples:
#   ./scripts/launch.sh 1 configs/tts_lite.yaml --train-steps 200
#   ./scripts/launch.sh 4 configs/tts_lite.yaml
#
# Uses torchrun, which sets RANK/WORLD_SIZE/LOCAL_RANK that train_lite reads.
set -euo pipefail

NGPU="${1:-1}"; shift || true
CONFIG="${1:-configs/tts_lite.yaml}"; shift || true

exec torchrun --standalone --nproc_per_node="${NGPU}" \
    -m latent_lm.train_lite --config "${CONFIG}" "$@"
