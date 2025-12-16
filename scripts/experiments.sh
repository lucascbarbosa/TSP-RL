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
#
# =============================================================================
# DESCOBERTAS PRINCIPAIS (resumo dos experimentos)
# =============================================================================
#
# 1. DELTA vs SPARSE REWARD
#    - Delta vence na maioria dos tamanhos (n=20, 40, 50, 70)
#    - Sparse só é melhor para n=30 (val_gap 1.75% vs 3.37%)
#    - Delta é mais robusto e consistente em geral
#
# 2. DOUBLE vs STANDARD DQN
#    - Double DQN melhor para n pequeno (20-30) e n grande (70)
#    - Standard DQN melhor para n intermediário (40-50)
#    - Q-overestimation do Standard é 2-4x maior que Double
#    - Q-overestimation não necessariamente prejudica performance
#
# 3. LEARNING RATE
#    - lr=0.0003 foi excepcional para n=20 (val_gap=0.10%)
#    - lr=0.0003 também ajuda Double DQN n=50 (4.87% vs 8.56%)
#    - Para n=70, lr=0.001 (default) é melhor
#
# 4. OUTROS HIPERPARÂMETROS
#    - hidden_dim=128: ajuda Double DQN (0.98% vs 1.30% para n=20)
#    - gamma=0.999: ajuda Standard DQN (0.75% vs 1.07% para n=20)
#    - episodes=400: não melhora, só aumenta Q-overestimation
#
# 5. ROBUSTEZ
#    - Variância de ~1% entre runs idênticas
#    - Tendências são consistentes apesar da variância
#
# 6. MELHOR CONFIG POR TAMANHO (EUC_2D)
#    - n=20: Standard DQN + delta + lr=0.0003 → 0.10%
#    - n=30: Double DQN + sparse             → 1.75%
#    - n=40: Standard DQN + delta            → 4.81%
#    - n=50: Standard DQN + delta            → 4.04%
#    - n=70: Double DQN + delta              → 2.69%
#
# =============================================================================

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

    echo
    echo "############################################################"
    echo "# Experimento: $name"
    echo "# Comando: ./scripts/pipeline.sh $@"
    echo "############################################################"
    echo

    if $DRY_RUN; then
        echo "[dry-run] Pulando execução"
        return
    fi

    local start=$(date +%s)
    ./scripts/pipeline.sh "$@"
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
# [DONE] 20251215_210025_EUC_2D_n20-30-40_ep200_delta
# NOTA: n=10 removido (trivialmente fácil, converge imediatamente para gap ~0%)
# run_experiment "1_dqn_vs_double" \
#     --types "EUC_2D" \
#     --sizes "20 30 40" \
#     --episodes 200

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
# [DONE] 20251215_211258_EUC_2D_n50-70_ep300_sparse
# NOTA: sparse teve val_gap 2.3% vs 5.8% (delta) para n=30
# run_experiment "4a_scalability_sparse" \
#     --types "EUC_2D" \
#     --sizes "50 70" \
#     --episodes 300 \
#     --time_budget 10.0 \
#     --reward_type sparse

# [DONE] 20251215_212533_EUC_2D_n50-70_ep300_delta
# Comparação direta delta vs sparse em instâncias grandes
# run_experiment "4b_scalability_delta" \
#     --types "EUC_2D" \
#     --sizes "50 70" \
#     --episodes 300 \
#     --time_budget 10.0 \
#     --reward_type delta

# -----------------------------------------------------------------------------
# 5. Ablação: Efeito do Epsilon
# -----------------------------------------------------------------------------
# [DONE] Redundante com 2a_reward_delta (mesmos parâmetros default)
# run_experiment "5a_epsilon_default" \
#     --types "EUC_2D" \
#     --sizes "20 30" \
#     --episodes 200 \
#     --epsilon_start 1.0 \
#     --epsilon_end 0.05

# [DONE] 20251215_213008_EUC_2D_n20-30_ep200_delta (ε=0.5→0.01)
# run_experiment "5b_epsilon_greedy_early" \
#     --types "EUC_2D" \
#     --sizes "20 30" \
#     --episodes 200 \
#     --epsilon_start 0.5 \
#     --epsilon_end 0.01

# [SKIP] Redundante - epsilon_end alto causa instabilidade sem benefício claro
# run_experiment "5c_epsilon_more_exploration" \
#     --types "EUC_2D" \
#     --sizes "20 30" \
#     --episodes 200 \
#     --epsilon_start 1.0 \
#     --epsilon_end 0.20

# =============================================================================
# 6. Experimentos Complementares (baseados em análise dos resultados anteriores)
# =============================================================================
# Descobertas até agora:
#   - Sparse reward é MUITO melhor para n=30 (val_gap 2.3% vs 5.8%)
#   - Para n=20, delta é levemente melhor
#   - Falta n=40 sparse para completar curva de escalabilidade
#   - Double DQN não melhora consistentemente val_gap, mas reduz Q-overestimation

# -----------------------------------------------------------------------------
# 6a. Sparse para n=40: Completa curva de escalabilidade
# -----------------------------------------------------------------------------
# [DONE] 20251215_220027_EUC_2D_n40_ep250_sparse
# Justificativa: Temos delta para n=40, mas não sparse. Crucial para comparação.
# run_experiment "6a_sparse_n40" \
#     --types "EUC_2D" \
#     --sizes "40" \
#     --episodes 250 \
#     --time_budget 8.0 \
#     --reward_type sparse

# -----------------------------------------------------------------------------
# 6b. Sparse para múltiplos tipos: Generalização do benefício sparse
# -----------------------------------------------------------------------------
# [DONE] 20251215_221609_ATT-GEO_n30-40_ep200_sparse
# Justificativa: Verificar se sparse também é melhor para ATT/GEO (n=30-40)
# run_experiment "6b_sparse_multitypes" \
#     --types "ATT GEO" \
#     --sizes "30 40" \
#     --episodes 200 \
#     --time_budget 8.0 \
#     --reward_type sparse \
#     --train_limit 300 \
#     --val_limit 80 \
#     --test_limit 80

# =============================================================================
# 7. Teste de Robustez: Mesma configuração, múltiplas runs
# =============================================================================
# Objetivo: Verificar se resultados são reprodutíveis (gaps médios, ações, etc.)
# Config: Double DQN + sparse, n=30 (onde sparse foi claramente melhor)

# [DONE] 20251215_225140_EUC_2D_n30_ep150_sparse
# run_experiment "7a_robustness_run1" \
#     --types "EUC_2D" \
#     --sizes "30" \
#     --episodes 150 \
#     --reward_type sparse

# [DONE] 20251215_225352_EUC_2D_n30_ep150_sparse
# run_experiment "7b_robustness_run2" \
#     --types "EUC_2D" \
#     --sizes "30" \
#     --episodes 150 \
#     --reward_type sparse

# [DONE] 20251215_225605_EUC_2D_n30_ep150_sparse
# run_experiment "7c_robustness_run3" \
#     --types "EUC_2D" \
#     --sizes "30" \
#     --episodes 150 \
#     --reward_type sparse

# =============================================================================
# 8. Ablação: Learning Rate
# =============================================================================
# Default: lr=0.001. Testar 0.0003 (mais conservador) e 0.003 (mais agressivo)
# n=20 para rapidez

# [DONE] 20251215_225808_EUC_2D_n20_ep150_delta
# run_experiment "8a_lr_low" \
#     --types "EUC_2D" \
#     --sizes "20" \
#     --episodes 150 \
#     --lr 0.0003 \
#     --reward_type delta

# [DONE] 20251215_230011_EUC_2D_n20_ep150_delta
# run_experiment "8b_lr_high" \
#     --types "EUC_2D" \
#     --sizes "20" \
#     --episodes 150 \
#     --lr 0.003 \
#     --reward_type delta

# =============================================================================
# 9. Ablação: Capacidade da Rede (hidden_dim)
# =============================================================================
# Default: hidden_dim=64. Testar 32 (menor) e 128 (maior)

# [DONE] 20251215_230213_EUC_2D_n20_ep150_delta
# run_experiment "9a_hidden_small" \
#     --types "EUC_2D" \
#     --sizes "20" \
#     --episodes 150 \
#     --hidden_dim 32 \
#     --reward_type delta

# [DONE] 20251215_230416_EUC_2D_n20_ep150_delta
# run_experiment "9b_hidden_large" \
#     --types "EUC_2D" \
#     --sizes "20" \
#     --episodes 150 \
#     --hidden_dim 128 \
#     --reward_type delta

# =============================================================================
# 10. Ablação: Número de Episódios (convergência)
# =============================================================================
# Verificar se mais episódios melhora ou se 200 é suficiente

# [DONE] 20251215_230602_EUC_2D_n20_ep100_delta
# run_experiment "10a_episodes_short" \
#     --types "EUC_2D" \
#     --sizes "20" \
#     --episodes 100 \
#     --reward_type delta

# [DONE] 20251215_230914_EUC_2D_n20_ep400_delta
# run_experiment "10b_episodes_long" \
#     --types "EUC_2D" \
#     --sizes "20" \
#     --episodes 400 \
#     --reward_type delta

# =============================================================================
# 11. Ablação: Gamma (discount factor)
# =============================================================================
# Default: gamma=0.99. Testar 0.95 (mais míope) e 0.999 (mais longe)

# [DONE] 20251215_231115_EUC_2D_n20_ep150_delta
# run_experiment "11a_gamma_low" \
#     --types "EUC_2D" \
#     --sizes "20" \
#     --episodes 150 \
#     --gamma 0.95 \
#     --reward_type delta

# [DONE] 20251215_231317_EUC_2D_n20_ep150_delta
# run_experiment "11b_gamma_high" \
#     --types "EUC_2D" \
#     --sizes "20" \
#     --episodes 150 \
#     --gamma 0.999 \
#     --reward_type delta

# =============================================================================
# 12. Teste de Learning Rate Baixa em Instâncias Maiores
# =============================================================================
# Descoberta: lr=0.0003 foi o melhor para n=20 (val_gap=0.10%)
# Objetivo: Verificar se lr baixa também melhora para n=50 e n=70
# Resultado: lr=0.0003 só ajuda Double DQN n=50 (4.87% vs 8.56%). Para n=70, lr=0.001 é melhor.

# [DONE] 20251215_233924_EUC_2D_n50_ep300_delta
# run_experiment "12a_lr_low_n50" \
#     --types "EUC_2D" \
#     --sizes "50" \
#     --episodes 300 \
#     --lr 0.0003 \
#     --time_budget 10.0 \
#     --reward_type delta

# [DONE] 20251215_234659_EUC_2D_n70_ep300_delta
# run_experiment "12b_lr_low_n70" \
#     --types "EUC_2D" \
#     --sizes "70" \
#     --episodes 300 \
#     --lr 0.0003 \
#     --time_budget 10.0 \
#     --reward_type delta

# #############################################################################
# #############################################################################
#
#                           RUN FINAL
#
# #############################################################################
# #############################################################################
#
# Configuração baseada nas descobertas dos experimentos anteriores:
#   - lr=0.0003: excepcional para n=20 (0.10%), bom para Double DQN n=50
#   - delta: mais robusto que sparse na maioria dos tamanhos
#   - Double DQN: melhor para n pequeno (20-30) e grande (70)
#   - Standard DQN: melhor para n intermediário (40-50)
#
# Esta run usa:
#   - Todas as instâncias de teste disponíveis (por tamanho)
#   - Mais instâncias de treino/validação
#   - Mais episódios para convergência
#   - Configuração otimizada (lr=0.0003)
#
# Tempo estimado: ~2-3 horas (depende do hardware)
# #############################################################################

# =============================================================================
# 13. Run Final: EUC_2D Completa
# =============================================================================
# Todos os tamanhos relevantes com config otimizada
# train_limit/val_limit/test_limit = 0 significa "usar todas disponíveis para o tamanho"

# [DONE] 20251216_005711_EUC_2D_n20-30-40-50-70_ep300_delta
# run_experiment "FINAL_EUC2D_all_sizes" \
#     --types "EUC_2D" \
#     --sizes "20 30 40 50 70" \
#     --episodes 300 \
#     --lr 0.0003 \
#     --time_budget 10.0 \
#     --reward_type delta \
#     --train_limit 800 \
#     --val_limit 200 \
#     --test_limit 200

# =============================================================================
# 14. Run Final: Generalização (ATT e GEO)
# =============================================================================
# Verificar se configuração otimizada generaliza para outros tipos

# [DONE] 20251216_012726_ATT-GEO_n20-30-40_ep250_delta
# run_experiment "FINAL_ATT_GEO" \
#     --types "ATT GEO" \
#     --sizes "20 30 40" \
#     --episodes 250 \
#     --lr 0.0003 \
#     --time_budget 8.0 \
#     --reward_type delta \
#     --train_limit 600 \
#     --val_limit 150 \
#     --test_limit 150

# =============================================================================
# 15. Run Final: Escalabilidade Extrema (n=100)
# =============================================================================
# Testar limite de escalabilidade do método

# [DONE] 20251216_021148_EUC_2D_n100_ep400_delta
# run_experiment "FINAL_n100" \
#     --types "EUC_2D" \
#     --sizes "100" \
#     --episodes 400 \
#     --lr 0.0003 \
#     --time_budget 20.0 \
#     --reward_type delta \
#     --train_limit 500 \
#     --val_limit 100 \
#     --test_limit 100

# =============================================================================
# 16. Run Final: Sparse para n=30 (onde foi melhor)
# =============================================================================
# Confirmação do melhor caso do sparse com mais dados

# [DONE] 20251216_021733_EUC_2D_n30_ep300_sparse
# run_experiment "FINAL_sparse_n30" \
#     --types "EUC_2D" \
#     --sizes "30" \
#     --episodes 300 \
#     --lr 0.0003 \
#     --time_budget 8.0 \
#     --reward_type sparse \
#     --train_limit 800 \
#     --val_limit 200 \
#     --test_limit 200

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
