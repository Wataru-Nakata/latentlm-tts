#!/usr/bin/env bash
# Convenience torchrun launcher for single-node training.
#
#   ./scripts/launch.sh [NGPU] [CONFIG] [-- extra train args]
#
# Examples:
#   ./scripts/launch.sh 1 configs/tts_smoke.yaml --train-steps 50
#   ./scripts/launch.sh 4 configs/tts_qwen_500m.yaml
#
# Requires the training stack on PYTHONPATH (transformer-engine, megatron-core,
# megatron-bridge). Easiest via the NeMo container (see examples/pbs/), but a
# pip env with those packages works too. torchrun sets RANK/WORLD_SIZE/LOCAL_RANK.
# For multi-node, use examples/pbs/train_tts.pbs (mpirun rendezvous).
set -euo pipefail

NGPU="${1:-1}"; shift || true
CONFIG="${1:-configs/tts_smoke.yaml}"; shift || true

exec torchrun --standalone --nproc_per_node="${NGPU}" \
    -m latent_lm.train --config "${CONFIG}" "$@"
