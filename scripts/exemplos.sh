#!/bin/bash
#
# Pipeline de exemplo para TSP-RL.
#
# Executa o pipeline completo em um subconjunto pequeno:
# - Tipos: EUC_2D, ATT, GEO
# - Tamanhos: 10, 20, 30
# - Limite: 100 instâncias por (tipo, tamanho)
#
# Usage:
#     ./scripts/exemplos.sh
#
# Tempo estimado: ~5-10 minutos dependendo do hardware.

set -e

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Verify we're in the right place
if [[ ! -f "CLAUDE.md" ]] || [[ ! -d "src" ]]; then
    echo "ERROR: Not in TSP-RL project root. Aborting."
    exit 1
fi

echo "========================================"
echo "TSP-RL Pipeline de Exemplo"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Configuration
TYPES="EUC_2D ATT GEO"
SIZES="10 20 30"
LIMIT=100
BETA=1.0

echo "Configuração:"
echo "  Tipos:   $TYPES"
echo "  Tamanhos: $SIZES"
echo "  Limite:  $LIMIT instâncias por (tipo, tamanho)"
echo "  Beta:    $BETA (desconto temporal)"
echo ""

# Step 1: Generate splits (if not exists)
echo "========================================"
echo "1. Gerando splits train/test..."
echo "========================================"
if [[ -f "data/splits.json" ]]; then
    echo "   data/splits.json já existe, pulando..."
else
    python scripts/generate_splits.py
fi
echo ""

# Step 2: Generate transitions
echo "========================================"
echo "2. Gerando transições (ILS com ações aleatórias)..."
echo "========================================"
for type in $TYPES; do
    echo "--- $type ---"
    python scripts/train_transitions.py \
        --split_path data/splits.json \
        --dataset_path "data/${type}.json" \
        --output_dir "data/train/${type}" \
        --sizes $SIZES \
        --limit $LIMIT \
        --beta $BETA
    echo ""
done

# Step 3: Train Q-tables
echo "========================================"
echo "3. Treinando Q-tables..."
echo "========================================"
python scripts/train_qtable.py \
    --types $TYPES \
    --sizes $SIZES
echo ""

# Step 4: Evaluate
echo "========================================"
echo "4. Avaliando Q-ILS nas instâncias de teste..."
echo "========================================"
python scripts/evaluate.py \
    --types $TYPES \
    --sizes $SIZES \
    --limit $LIMIT \
    --output data/results/results_exemplo.csv
echo ""

echo "========================================"
echo "Pipeline concluído!"
echo "========================================"
echo "Resultados em: data/results/results_exemplo.csv"
echo "Q-tables em:   data/q_tables/"
echo "Plots em:      data/plots/"
