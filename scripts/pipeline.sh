#!/bin/bash
#
# DQN-ILS Pipeline wrapper - invoca pipeline.py e organiza outputs.
#
# Usage:
#     ./scripts/pipeline.sh                         # defaults
#     ./scripts/pipeline.sh --types "EUC_2D ATT" --sizes "10 20"
#
# Outputs são movidos para experiments/<timestamp>_<params>/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Defaults (must match pipeline.py argparse defaults)
TYPES="EUC_2D"
SIZES="10 20"
EPISODES=200
TIME_BUDGET=5.0
GAMMA=0.99
LR=0.001
HIDDEN_DIM=64
HISTORY_LEN=1
EPSILON_START=1.0
EPSILON_END=0.05
REWARD_TYPE="delta"
TRAIN_LIMIT=100
VAL_LIMIT=40
TEST_LIMIT=40
BASELINE=""
COMPARE_DOUBLE=""
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
        --epsilon_start) EPSILON_START="$2"; shift 2 ;;
        --epsilon_end)  EPSILON_END="$2"; shift 2 ;;
        --reward_type)  REWARD_TYPE="$2"; shift 2 ;;
        --train_limit)  TRAIN_LIMIT="$2"; shift 2 ;;
        --val_limit)    VAL_LIMIT="$2"; shift 2 ;;
        --test_limit)   TEST_LIMIT="$2"; shift 2 ;;
        --no_baseline)  BASELINE="--no_baseline"; shift ;;
        --no_compare_double) COMPARE_DOUBLE="--no_compare_double"; shift ;;
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
python scripts/pipeline.py \
    --types $TYPES \
    --sizes $SIZES \
    --episodes "$EPISODES" \
    --time_budget "$TIME_BUDGET" \
    --gamma "$GAMMA" \
    --lr "$LR" \
    --hidden_dim "$HIDDEN_DIM" \
    --history_len "$HISTORY_LEN" \
    --epsilon_start "$EPSILON_START" \
    --epsilon_end "$EPSILON_END" \
    --reward_type "$REWARD_TYPE" \
    --train_limit "$TRAIN_LIMIT" \
    --val_limit "$VAL_LIMIT" \
    --test_limit "$TEST_LIMIT" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    $BASELINE \
    $COMPARE_DOUBLE

# =============================================================================
# Organize outputs into experiment folder
# =============================================================================

# Build folder name: YYYYMMDD_HHMMSS_<types>_n<sizes>_ep<episodes>_<reward>
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TYPES_SLUG=$(echo "$TYPES" | tr ' ' '-')
SIZES_SLUG=$(echo "$SIZES" | tr ' ' '-')
EXP_NAME="${TIMESTAMP}_${TYPES_SLUG}_n${SIZES_SLUG}_ep${EPISODES}_${REWARD_TYPE}"
EXP_DIR="experiments/${EXP_NAME}"

echo
echo "=================================================="
echo "Organizing outputs to: $EXP_DIR"
echo "=================================================="

mkdir -p "$EXP_DIR"

# Save parameters as JSON
cat > "$EXP_DIR/params.json" << EOF
{
  "timestamp": "$TIMESTAMP",
  "types": "$TYPES",
  "sizes": "$SIZES",
  "episodes": $EPISODES,
  "time_budget": $TIME_BUDGET,
  "gamma": $GAMMA,
  "lr": $LR,
  "hidden_dim": $HIDDEN_DIM,
  "history_len": $HISTORY_LEN,
  "epsilon_start": $EPSILON_START,
  "epsilon_end": $EPSILON_END,
  "reward_type": "$REWARD_TYPE",
  "train_limit": $TRAIN_LIMIT,
  "val_limit": $VAL_LIMIT,
  "test_limit": $TEST_LIMIT,
  "baseline": $([ -z "$BASELINE" ] && echo "true" || echo "false"),
  "compare_double": $([ -z "$COMPARE_DOUBLE" ] && echo "true" || echo "false"),
  "device": "$DEVICE",
  "workers": $WORKERS
}
EOF

# Move generated files
if [ -d "models/dqn" ] && [ "$(ls -A models/dqn 2>/dev/null)" ]; then
    mv models/dqn "$EXP_DIR/models"
    mkdir -p models/dqn  # recreate empty dir
    echo "  Moved: models/dqn/ -> $EXP_DIR/models/"
fi

if [ -d "data/results" ] && [ "$(ls -A data/results 2>/dev/null)" ]; then
    mv data/results "$EXP_DIR/results"
    mkdir -p data/results
    echo "  Moved: data/results/ -> $EXP_DIR/results/"
fi

if [ -d "data/plots" ] && [ "$(ls -A data/plots 2>/dev/null)" ]; then
    mv data/plots "$EXP_DIR/plots"
    mkdir -p data/plots
    echo "  Moved: data/plots/ -> $EXP_DIR/plots/"
fi

echo
echo "Experiment saved to: $EXP_DIR"
echo "  params.json - experiment configuration"
echo "  models/     - trained models and stats"
echo "  results/    - evaluation CSVs"
echo "  plots/      - generated plots"
