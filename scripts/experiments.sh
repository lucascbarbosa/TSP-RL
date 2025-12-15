#!/bin/bash
#
# Executa bateria de experimentos para apresentação/relatório.
#
# Tempo estimado: ~2 horas (16 workers, CPU)
#
# Usage:
#     ./scripts/experiments.sh           # Executa todos
#     ./scripts/experiments.sh --dry-run # Mostra comandos sem executar
#
# Outputs vão para experiments/<timestamp>_<params>/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
    echo
fi

# Timing
TOTAL_START=$(date +%s)
declare -A TIMINGS

run_experiment() {
    local name="$1"
    shift
    local args="$@"

    echo
    echo "############################################################"
    echo "# Experimento: $name"
    echo "# Comando: ./scripts/pipeline.sh $args"
    echo "############################################################"
    echo

    if $DRY_RUN; then
        echo "[dry-run] Pulando execução"
        return
    fi

    local start=$(date +%s)
    ./scripts/pipeline.sh $args
    local end=$(date +%s)
    local duration=$((end - start))
    TIMINGS["$name"]=$duration

    echo
    echo ">>> $name concluído em $((duration / 60))m $((duration % 60))s"
    echo
}

echo "============================================================"
echo "  BATERIA DE EXPERIMENTOS DQN-ILS"
echo "  Início: $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# 1. Experimento Principal: DQN vs Double DQN
# -----------------------------------------------------------------------------
run_experiment "1_dqn_vs_double" \
    --types "EUC_2D" \
    --sizes "10 20 30" \
    --episodes 200

# -----------------------------------------------------------------------------
# 2. Comparação de Reward Types
# -----------------------------------------------------------------------------
# [DONE] 20251215_193744_EUC_2D_n20-30_ep200_delta
# run_experiment "2a_reward_delta" \
#     --types "EUC_2D" \
#     --sizes "20 30" \
#     --episodes 200 \
#     --reward_type delta

# [DONE] 20251215_195021_EUC_2D_n20-30_ep200_sparse
# run_experiment "2b_reward_sparse" \
#     --types "EUC_2D" \
#     --sizes "20 30" \
#     --episodes 200 \
#     --reward_type sparse

# -----------------------------------------------------------------------------
# 3. Generalização por Tipo de Instância
# -----------------------------------------------------------------------------
# [DONE] EUC_2D ATT GEO, sizes 20 40, train_limit 500
# run_experiment "3_generalization_types" \
#     --types "EUC_2D ATT GEO" \
#     --sizes "20 40" \
#     --episodes 200 \
#     --train_limit 500 \
#     --val_limit 100 \
#     --test_limit 100

# -----------------------------------------------------------------------------
# 4. Escalabilidade (Instâncias Maiores)
# -----------------------------------------------------------------------------
run_experiment "4_scalability" \
    --types "EUC_2D" \
    --sizes "40 50 70" \
    --episodes 250 \
    --time_budget 8.0

# -----------------------------------------------------------------------------
# 5. Ablação: Efeito do Epsilon
# -----------------------------------------------------------------------------
run_experiment "5a_epsilon_default" \
    --types "EUC_2D" \
    --sizes "20 30" \
    --episodes 200 \
    --epsilon_start 1.0 \
    --epsilon_end 0.05

run_experiment "5b_epsilon_less_exploration" \
    --types "EUC_2D" \
    --sizes "20 30" \
    --episodes 200 \
    --epsilon_start 0.5 \
    --epsilon_end 0.01

run_experiment "5c_epsilon_more_exploration" \
    --types "EUC_2D" \
    --sizes "20 30" \
    --episodes 200 \
    --epsilon_start 1.0 \
    --epsilon_end 0.20

# -----------------------------------------------------------------------------
# Resumo
# -----------------------------------------------------------------------------
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo
echo "============================================================"
echo "  RESUMO DA BATERIA DE EXPERIMENTOS"
echo "  Fim: $(date)"
echo "  Duração total: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m $((TOTAL_DURATION % 60))s"
echo "============================================================"
echo

if ! $DRY_RUN; then
    echo "Tempos por experimento:"
    for exp in "${!TIMINGS[@]}"; do
        duration=${TIMINGS[$exp]}
        printf "  %-30s %3dm %02ds\n" "$exp" $((duration / 60)) $((duration % 60))
    done | sort
    echo
fi

echo "Resultados em: experiments/"
ls -dt experiments/*/ 2>/dev/null | head -10

echo
echo "Próximos passos:"
echo "  1. Revisar plots em experiments/*/plots/"
echo "  2. Comparar stats em experiments/*/models/*_stats.json"
echo "  3. Analisar CSVs em experiments/*/results/"
