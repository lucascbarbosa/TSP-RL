#!/bin/bash
#
# DQN-ILS Pipeline: train and evaluate models.
#
# Runs the full pipeline with explicit hyperparameters for easy experimentation.
# Default configuration is minimal for quick testing (~5-10 min).
#
# Usage:
#     ./scripts/pipeline.sh                    # Use defaults
#     ./scripts/pipeline.sh --types "EUC_2D ATT" --sizes "10 20"
#     ./scripts/pipeline.sh --episodes 500 --limit 50
#
# For full training, increase episodes and remove limit:
#     ./scripts/pipeline.sh --episodes 2000 --limit ""

set -e

# =============================================================================
# Configuration (edit these for experiments)
# =============================================================================

# Instance selection
TYPES="EUC_2D"              # Space-separated: "EUC_2D ATT GEO"
SIZES="10 20"               # Space-separated: "10 20 30 50 100"

# Training hyperparameters
EPISODES=200                # Number of training episodes (default: 2000)
TIME_BUDGET=10.0            # Base time budget in seconds
GAMMA=0.99                  # Discount factor
LR=0.001                    # Learning rate
HIDDEN_DIM=64               # Hidden layer dimension
HISTORY_LEN=2               # Past actions in state

# Instance limits (for quick testing, set to "" for full dataset)
TRAIN_LIMIT=100             # Limit training instances per size
EVAL_LIMIT=20               # Limit evaluation instances per size

# Evaluation
BASELINE=true               # Include GRASP+2opt baseline (same time budget as DQN)

# Device
DEVICE="cpu"                # "cpu" or "cuda"

# =============================================================================
# Parse command line arguments (override defaults)
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --types)
            TYPES="$2"
            shift 2
            ;;
        --sizes)
            SIZES="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --time_budget)
            TIME_BUDGET="$2"
            shift 2
            ;;
        --gamma)
            GAMMA="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --history_len)
            HISTORY_LEN="$2"
            shift 2
            ;;
        --train_limit)
            TRAIN_LIMIT="$2"
            shift 2
            ;;
        --eval_limit)
            EVAL_LIMIT="$2"
            shift 2
            ;;
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --types TYPES         Instance types (default: \"$TYPES\")"
            echo "  --sizes SIZES         Instance sizes (default: \"$SIZES\")"
            echo "  --episodes N          Training episodes (default: $EPISODES)"
            echo "  --time_budget T       Base time budget in seconds (default: $TIME_BUDGET)"
            echo "  --gamma G             Discount factor (default: $GAMMA)"
            echo "  --lr LR               Learning rate (default: $LR)"
            echo "  --hidden_dim D        Hidden layer dimension (default: $HIDDEN_DIM)"
            echo "  --history_len H       Past actions in state (default: $HISTORY_LEN)"
            echo "  --train_limit N       Limit training instances (default: $TRAIN_LIMIT)"
            echo "  --eval_limit N        Limit evaluation instances (default: $EVAL_LIMIT)"
            echo "  --baseline true|false Include baseline comparison (default: $BASELINE)"
            echo "  --device cpu|cuda     Device for training (default: $DEVICE)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f "CLAUDE.md" ]] || [[ ! -d "src" ]]; then
    echo "ERROR: Not in TSP-RL project root. Aborting."
    exit 1
fi

echo "========================================"
echo "DQN-ILS Pipeline"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo ""
echo "Configuration:"
echo "  Types:       $TYPES"
echo "  Sizes:       $SIZES"
echo "  Episodes:    $EPISODES"
echo "  Time budget: $TIME_BUDGET s"
echo "  Gamma:       $GAMMA"
echo "  LR:          $LR"
echo "  Hidden dim:  $HIDDEN_DIM"
echo "  History len: $HISTORY_LEN"
echo "  Train limit: ${TRAIN_LIMIT:-\"(no limit)\"}"
echo "  Eval limit:  ${EVAL_LIMIT:-\"(no limit)\"}"
echo "  Baseline:    $BASELINE"
echo "  Device:      $DEVICE"
echo ""

# =============================================================================
# Step 1: Generate splits (if not exists)
# =============================================================================

echo "========================================"
echo "1. Checking splits..."
echo "========================================"

if [[ -f "data/splits.json" ]]; then
    echo "   data/splits.json exists, skipping generation."
else
    echo "   Generating splits..."
    python scripts/generate_splits.py --seed 42 --train_ratio 0.9
fi
echo ""

# =============================================================================
# Step 2: Train models
# =============================================================================

echo "========================================"
echo "2. Training DQN models..."
echo "========================================"

# Build limit argument
LIMIT_ARG=""
if [[ -n "$TRAIN_LIMIT" ]]; then
    LIMIT_ARG="--limit $TRAIN_LIMIT"
fi

for TYPE in $TYPES; do
    echo ""
    echo "--- Training $TYPE ---"
    python scripts/train_dqn.py \
        --type "$TYPE" \
        --sizes $SIZES \
        --episodes "$EPISODES" \
        --time_budget "$TIME_BUDGET" \
        --gamma "$GAMMA" \
        --lr "$LR" \
        --hidden_dim "$HIDDEN_DIM" \
        --history_len "$HISTORY_LEN" \
        --device "$DEVICE" \
        $LIMIT_ARG
done
echo ""

# =============================================================================
# Step 3: Evaluate models
# =============================================================================

echo "========================================"
echo "3. Evaluating models..."
echo "========================================"

# Build limit and baseline arguments
EVAL_LIMIT_ARG=""
if [[ -n "$EVAL_LIMIT" ]]; then
    EVAL_LIMIT_ARG="--limit $EVAL_LIMIT"
fi

BASELINE_ARG=""
if [[ "$BASELINE" == "true" ]]; then
    BASELINE_ARG="--baseline"
fi

for TYPE in $TYPES; do
    echo ""
    echo "--- Evaluating $TYPE ---"
    python scripts/evaluate_dqn.py \
        --model "models/dqn/${TYPE}_*.pt" \
        --time_budget "$TIME_BUDGET" \
        --history_len "$HISTORY_LEN" \
        --hidden_dim "$HIDDEN_DIM" \
        --device "$DEVICE" \
        $EVAL_LIMIT_ARG \
        $BASELINE_ARG
done
echo ""

# =============================================================================
# Step 4: Generate plots
# =============================================================================

echo "========================================"
echo "4. Generating plots..."
echo "========================================"

python scripts/generate_plots.py \
    --models "models/dqn/*.pt" \
    --results "data/results/*.csv" \
    --history_len "$HISTORY_LEN" \
    --hidden_dim "$HIDDEN_DIM"
echo ""

# =============================================================================
# Done
# =============================================================================

echo "========================================"
echo "Pipeline complete!"
echo "========================================"
echo ""
echo "Outputs:"
echo "  Models:  models/dqn/"
echo "  Results: data/results/"
echo "  Plots:   data/plots/"
