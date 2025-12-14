#!/bin/bash
#
# DQN-ILS Pipeline wrapper - invoca pipeline.py com os parâmetros recebidos.
#
# Usage:
#     ./scripts/pipeline.sh                         # defaults
#     ./scripts/pipeline.sh --types "EUC_2D ATT" --sizes "10 20"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Defaults
TYPES="EUC_2D"
SIZES="10 20"
EPISODES=64
TIME_BUDGET=5.0
GAMMA=0.99
LR=0.001
HIDDEN_DIM=64
HISTORY_LEN=1
TRAIN_LIMIT=100
EVAL_LIMIT=20
BASELINE=""
DEVICE="cpu"
WORKERS=16

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --types)        TYPES="$2"; shift 2 ;;
        --sizes)        SIZES="$2"; shift 2 ;;
        --episodes)     EPISODES="$2"; shift 2 ;;
        --time_budget)  TIME_BUDGET="$2"; shift 2 ;;
        --gamma)        GAMMA="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --hidden_dim)   HIDDEN_DIM="$2"; shift 2 ;;
        --history_len)  HISTORY_LEN="$2"; shift 2 ;;
        --train_limit)  TRAIN_LIMIT="$2"; shift 2 ;;
        --eval_limit)   EVAL_LIMIT="$2"; shift 2 ;;
        --no_baseline)  BASELINE="--no_baseline"; shift ;;
        --device)       DEVICE="$2"; shift 2 ;;
        --workers)      WORKERS="$2"; shift 2 ;;
        --help|-h)
            python scripts/pipeline.py --help
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Invoke pipeline.py
exec python scripts/pipeline.py \
    --types $TYPES \
    --sizes $SIZES \
    --episodes "$EPISODES" \
    --time_budget "$TIME_BUDGET" \
    --gamma "$GAMMA" \
    --lr "$LR" \
    --hidden_dim "$HIDDEN_DIM" \
    --history_len "$HISTORY_LEN" \
    --train_limit "$TRAIN_LIMIT" \
    --eval_limit "$EVAL_LIMIT" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    $BASELINE
