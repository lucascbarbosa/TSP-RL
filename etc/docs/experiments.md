# Experimentos para Apresentação

Sugestões de experimentos para gerar resultados. Tempo estimado considera 16 workers em CPU.

> **Nota:** O `pipeline.sh` organiza automaticamente os outputs em `experiments/<timestamp>_<params>/`, incluindo um `params.json` com a configuração completa. Não é necessário renomear arquivos entre execuções.

## Execução Automatizada

Para rodar a bateria completa de experimentos (~2h):

```bash
./scripts/experiments.sh           # Executa todos
./scripts/experiments.sh --dry-run # Mostra comandos sem executar
```

O script `experiments.sh` executa os experimentos abaixo em sequência. Comente/descomente conforme necessário.

---

## 1. Experimento Principal: DQN vs Double DQN (EUC_2D)

Comparação central do trabalho. Gera curvas de aprendizado, Q-values e distribuição de ações.

```bash
./scripts/pipeline.sh --types "EUC_2D" --sizes "10 20 30" --episodes 200
```

**Tempo estimado:** ~15-25 min
**Outputs:** Plots comparativos em `data/plots/`, modelos em `models/dqn/`

---

## 2. Comparação de Reward Types: Delta vs Sparse

Testar se recompensa esparsa (só no final) compete com delta (melhoria incremental).

```bash
# Delta (padrão) - já executado no exp. 1, ou:
./scripts/pipeline.sh --types "EUC_2D" --sizes "20 30" --episodes 200 --reward_type delta

# Sparse
./scripts/pipeline.sh --types "EUC_2D" --sizes "20 30" --episodes 200 --reward_type sparse
```

**Tempo estimado:** ~10 min cada
**Nota:** Cada execução gera pasta separada em `experiments/`. Comparar stats JSONs entre pastas.

---

## 3. Generalização por Tipo de Instância

Verificar se o método funciona em diferentes métricas de distância.

```bash
./scripts/pipeline.sh --types "EUC_2D ATT GEO" --sizes "20 40" --episodes 200 --train_limit 500 --val_limit 100 --test_limit 100
```

**Tempo estimado:** ~25-35 min
**Outputs:** Resultados separados por tipo em `data/results/`

---

## 4. Escalabilidade (Instâncias Maiores)

Testar em instâncias maiores para mostrar escalabilidade. **Mais demorado.**

```bash
./scripts/pipeline.sh --types "EUC_2D" --sizes "30 50 70" --episodes 300 --time_budget 10.0
```

**Tempo estimado:** ~45-60 min
**Nota:** `time_budget` maior para instâncias grandes (escala com n²)

---

## 5. Experimento Rápido (Smoke Test)

Para validar que tudo funciona antes de rodar experimentos longos.

```bash
./scripts/pipeline.sh --types "EUC_2D" --sizes "10 20" --episodes 50 --train_limit 20 --val_limit 10 --test_limit 10
```

**Tempo estimado:** ~2-3 min

---

## 6. Ablação: Efeito do Epsilon

Comparar diferentes taxas de exploração.

```bash
# Exploração padrão (1.0 → 0.05)
./scripts/pipeline.sh --types "EUC_2D" --sizes "20 30" --episodes 200

# Menos exploração (0.5 → 0.01)
./scripts/pipeline.sh --types "EUC_2D" --sizes "20 30" --episodes 200 --epsilon_start 0.5 --epsilon_end 0.01

# Mais exploração final (1.0 → 0.2)
./scripts/pipeline.sh --types "EUC_2D" --sizes "20 30" --episodes 200 --epsilon_end 0.2
```

**Tempo estimado:** ~10 min cada

---

## Priorização Sugerida

Para uma apresentação com tempo limitado:

| Prioridade | Experimento | Justificativa |
|------------|-------------|---------------|
| 1 | Exp. 5 (smoke test) | Validar setup |
| 2 | Exp. 1 (DQN vs Double) | Resultado central |
| 3 | Exp. 3 (tipos) | Generalização |
| 4 | Exp. 2 (rewards) | Feature nova |
| 5 | Exp. 4 (escala) | Se houver tempo |

---

## Notas

- **Organização automática:** Outputs vão para `experiments/<timestamp>_<tipo>_n<sizes>_ep<eps>_<reward>/`
- **Estrutura por experimento:**
  ```
  experiments/20251215_193000_EUC_2D_n20-30_ep200_delta/
  ├── params.json   # configuração completa
  ├── models/       # *.pt, *_stats.json
  ├── results/      # *.csv
  └── plots/        # *.png
  ```
- **Paralelização:** `--workers 16` é o padrão, ajustar conforme CPU disponível
- **GPU:** Adicionar `--device cuda` se disponível (pouco ganho para redes pequenas)
- **Reprodutibilidade:** Splits são fixos (seed=42), mas treinamento tem estocasticidade
