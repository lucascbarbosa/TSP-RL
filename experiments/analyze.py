#!/usr/bin/env python3
"""
Análise de resultados dos experimentos DQN-ILS.

Usage:
    python experiments/analyze.py           # Tabela resumo
    python experiments/analyze.py --detail  # Detalhes por experimento
    python experiments/analyze.py --compare # Comparações específicas
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent


def load_experiments():
    """Load all experiment data."""
    results = []

    for exp_dir in sorted(SCRIPT_DIR.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        params_file = exp_dir / "params.json"
        if not params_file.exists():
            continue

        with open(params_file) as f:
            params = json.load(f)

        models_dir = exp_dir / "models"
        if not models_dir.exists():
            continue

        stats_files = list(models_dir.glob("*_stats.json"))

        for sf in sorted(stats_files):
            with open(sf) as f:
                stats = json.load(f)

            # q_values_mean is a list (per episode), get final value
            q_mean_raw = stats.get("q_values_mean")
            if isinstance(q_mean_raw, list) and q_mean_raw:
                q_mean = q_mean_raw[-1]  # last episode
            elif isinstance(q_mean_raw, (int, float)):
                q_mean = q_mean_raw
            else:
                q_mean = None

            results.append(
                {
                    "exp_dir": exp_dir.name,
                    "exp_short": exp_dir.name[9:],  # remove timestamp
                    "reward_type": params.get("reward_type", "delta"),
                    "type": stats.get("type", "?"),
                    "size": stats.get("size", 0),
                    "use_double": stats.get("use_double_dqn", False),
                    "epsilon_start": params.get("epsilon_start", 1.0),
                    "epsilon_end": params.get("epsilon_end", 0.05),
                    "val_gap": stats.get("val_avg_gap"),
                    "val_std": stats.get("val_std_gap"),
                    "test_gap": stats.get("final_avg_gap"),
                    "q_mean": q_mean,
                    "n_episodes": stats.get("n_episodes", 0),
                    "time_budget": stats.get("time_budget", 0),
                }
            )

    return results


def print_summary_table(results):
    """Print summary table of all results."""
    print()
    print("=" * 110)
    print("RESUMO DOS EXPERIMENTOS")
    print("=" * 110)
    print(
        f"{'Experimento':<35} {'Type':<7} {'n':>3} {'Dbl':<3} {'Reward':<7} {'ε':<10} {'Val%':>8} {'Test%':>8} {'Q_mean':>8}"
    )
    print("-" * 110)

    for r in results:
        eps = f"{r['epsilon_start']:.1f}→{r['epsilon_end']:.2f}"
        val = f"{r['val_gap']:.2f}" if r["val_gap"] is not None else "N/A"
        test = f"{r['test_gap']:.2f}" if r["test_gap"] is not None else "N/A"
        q = f"{r['q_mean']:.3f}" if r["q_mean"] is not None else "N/A"
        dbl = "Y" if r["use_double"] else "N"

        print(
            f"{r['exp_short']:<35} {r['type']:<7} {r['size']:>3} {dbl:<3} {r['reward_type']:<7} {eps:<10} {val:>8} {test:>8} {q:>8}"
        )

    print("=" * 110)
    print()


def compare_dqn_vs_double(results):
    """Compare DQN vs Double DQN."""
    print()
    print("=" * 80)
    print("COMPARAÇÃO: DQN vs Double DQN")
    print("=" * 80)

    # Group by (type, size, reward, epsilon)
    groups = defaultdict(list)
    for r in results:
        key = (r["type"], r["size"], r["reward_type"], r["epsilon_start"], r["epsilon_end"])
        groups[key].append(r)

    print(f"{'Type':<7} {'n':>3} {'Reward':<7} {'DQN_val%':>10} {'DDqn_val%':>10} {'Δ':>8} {'DQN_Q':>8} {'DDqn_Q':>8}")
    print("-" * 80)

    for key, grp in sorted(groups.items()):
        if len(grp) < 2:
            continue

        dqn = [r for r in grp if not r["use_double"]]
        ddqn = [r for r in grp if r["use_double"]]

        if not dqn or not ddqn:
            continue

        dqn = dqn[0]
        ddqn = ddqn[0]

        dqn_val = dqn["val_gap"] if dqn["val_gap"] is not None else float("nan")
        ddqn_val = ddqn["val_gap"] if ddqn["val_gap"] is not None else float("nan")
        delta = dqn_val - ddqn_val

        dqn_q = f"{dqn['q_mean']:.3f}" if dqn["q_mean"] else "N/A"
        ddqn_q = f"{ddqn['q_mean']:.3f}" if ddqn["q_mean"] else "N/A"

        print(
            f"{key[0]:<7} {key[1]:>3} {key[2]:<7} {dqn_val:>10.2f} {ddqn_val:>10.2f} {delta:>+8.2f} {dqn_q:>8} {ddqn_q:>8}"
        )

    print()


def compare_reward_types(results):
    """Compare delta vs sparse rewards."""
    print()
    print("=" * 80)
    print("COMPARAÇÃO: Delta vs Sparse Reward")
    print("=" * 80)

    # Group by (type, size, double, epsilon)
    groups = defaultdict(list)
    for r in results:
        key = (r["type"], r["size"], r["use_double"])
        groups[key].append(r)

    print(f"{'Type':<7} {'n':>3} {'Dbl':<3} {'Delta_val%':>11} {'Sparse_val%':>12} {'Δ':>8}")
    print("-" * 60)

    for key, grp in sorted(groups.items()):
        delta_rs = [r for r in grp if r["reward_type"] == "delta" and r["epsilon_start"] == 1.0]
        sparse_rs = [r for r in grp if r["reward_type"] == "sparse"]

        if not delta_rs or not sparse_rs:
            continue

        delta_r = delta_rs[0]
        sparse_r = sparse_rs[0]

        delta_val = delta_r["val_gap"] if delta_r["val_gap"] is not None else float("nan")
        sparse_val = sparse_r["val_gap"] if sparse_r["val_gap"] is not None else float("nan")
        diff = delta_val - sparse_val

        dbl = "Y" if key[2] else "N"
        print(f"{key[0]:<7} {key[1]:>3} {dbl:<3} {delta_val:>11.2f} {sparse_val:>12.2f} {diff:>+8.2f}")

    print()


def compare_epsilon(results):
    """Compare different epsilon schedules."""
    print()
    print("=" * 80)
    print("COMPARAÇÃO: Epsilon Schedules")
    print("=" * 80)

    # Group by (type, size, reward, double)
    groups = defaultdict(list)
    for r in results:
        if r["reward_type"] == "delta":
            key = (r["type"], r["size"], r["use_double"])
            groups[key].append(r)

    print(f"{'Type':<7} {'n':>3} {'Dbl':<3} {'ε=1→0.05':>10} {'ε=0.5→0.01':>12} {'Δ':>8}")
    print("-" * 60)

    for key, grp in sorted(groups.items()):
        default_eps = [r for r in grp if r["epsilon_start"] == 1.0 and r["epsilon_end"] == 0.05]
        greedy_eps = [r for r in grp if r["epsilon_start"] == 0.5 and r["epsilon_end"] == 0.01]

        if not default_eps or not greedy_eps:
            continue

        default_r = default_eps[0]
        greedy_r = greedy_eps[0]

        default_val = default_r["val_gap"] if default_r["val_gap"] is not None else float("nan")
        greedy_val = greedy_r["val_gap"] if greedy_r["val_gap"] is not None else float("nan")
        diff = default_val - greedy_val

        dbl = "Y" if key[2] else "N"
        print(f"{key[0]:<7} {key[1]:>3} {dbl:<3} {default_val:>10.2f} {greedy_val:>12.2f} {diff:>+8.2f}")

    print()


def scalability_analysis(results):
    """Analyze scalability by instance size."""
    print()
    print("=" * 80)
    print("ANÁLISE DE ESCALABILIDADE (por tamanho)")
    print("=" * 80)

    # Group by size
    by_size = defaultdict(list)
    for r in results:
        if r["type"] == "EUC_2D" and r["use_double"] and r["epsilon_start"] == 1.0:
            by_size[r["size"]].append(r)

    print(f"{'n':>3} {'Reward':<7} {'Val%':>8} {'Test%':>8} {'Q_mean':>8}")
    print("-" * 50)

    for size in sorted(by_size.keys()):
        for r in sorted(by_size[size], key=lambda x: x["reward_type"]):
            val = f"{r['val_gap']:.2f}" if r["val_gap"] is not None else "N/A"
            test = f"{r['test_gap']:.2f}" if r["test_gap"] is not None else "N/A"
            q = f"{r['q_mean']:.3f}" if r["q_mean"] is not None else "N/A"
            print(f"{size:>3} {r['reward_type']:<7} {val:>8} {test:>8} {q:>8}")

    print()


def type_generalization(results):
    """Analyze generalization across instance types."""
    print()
    print("=" * 80)
    print("GENERALIZAÇÃO POR TIPO DE INSTÂNCIA")
    print("=" * 80)

    # Filter for generalization experiment
    gen_results = [r for r in results if "ATT" in r["exp_dir"] or "GEO" in r["exp_dir"]]

    if not gen_results:
        print("Nenhum experimento de generalização encontrado.")
        return

    print(f"{'Type':<7} {'n':>3} {'Dbl':<3} {'Val%':>8} {'Test%':>8}")
    print("-" * 40)

    for r in sorted(gen_results, key=lambda x: (x["type"], x["size"], not x["use_double"])):
        dbl = "Y" if r["use_double"] else "N"
        val = f"{r['val_gap']:.2f}" if r["val_gap"] is not None else "N/A"
        test = f"{r['test_gap']:.2f}" if r["test_gap"] is not None else "N/A"
        print(f"{r['type']:<7} {r['size']:>3} {dbl:<3} {val:>8} {test:>8}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Análise de experimentos DQN-ILS")
    parser.add_argument("--detail", action="store_true", help="Mostrar detalhes")
    parser.add_argument("--compare", action="store_true", help="Mostrar comparações")
    parser.add_argument("--all", action="store_true", help="Mostrar tudo")
    args = parser.parse_args()

    results = load_experiments()

    if not results:
        print("Nenhum experimento encontrado em experiments/")
        return

    # Always show summary
    print_summary_table(results)

    if args.compare or args.all:
        compare_dqn_vs_double(results)
        compare_reward_types(results)
        compare_epsilon(results)

    if args.detail or args.all:
        scalability_analysis(results)
        type_generalization(results)


if __name__ == "__main__":
    main()
