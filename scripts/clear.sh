#!/bin/bash
#
# Clear generated files from TSP-RL project.
#
# Usage:
#     ./scripts/clear.sh          # Dry-run (shows what would be deleted)
#     ./scripts/clear.sh --force  # Actually delete files
#
# Can be invoked from any directory.

set -e

# Navigate to project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Verify we're in the right place
if [[ ! -f "CLAUDE.md" ]] || [[ ! -d "src" ]]; then
    echo "ERROR: Not in TSP-RL project root. Aborting."
    exit 1
fi

echo "Project root: $PROJECT_ROOT"
echo ""

# Files/directories to clean
TARGETS=(
    "data/splits.json"       # generate_splits.py output
    "data/results"           # Evaluation CSVs
    "data/plots"             # Generated plots
    "models/dqn"             # Trained DQN models
)

# Check for --force flag
FORCE=false
if [[ "$1" == "--force" ]] || [[ "$1" == "-f" ]]; then
    FORCE=true
fi

# List or delete targets
found=false
for target in "${TARGETS[@]}"; do
    if [[ -e "$target" ]]; then
        found=true
        if $FORCE; then
            echo "Removing: $target"
            rm -rf "$target"
        else
            if [[ -d "$target" ]]; then
                echo "[DIR]  $target"
            else
                echo "[FILE] $target"
            fi
        fi
    fi
done

if ! $found; then
    echo "Nothing to clean."
    exit 0
fi

if ! $FORCE; then
    echo ""
    echo "Dry-run complete. Use --force to actually delete."
fi
