"""Generate benchmark figures.

Produces, under ``docs/results/figures/``:

* one grouped bar chart per game (every strategy scored against each P1 opponent),
* a Minimax rollout-budget sweep (PALS mean reward vs K) — the paper's Fig 5.

    pip install -e ".[viz]"
    python -m scripts.plot_benchmarks

Deterministic given the fixed seed.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pals.bench.evaluate import evaluate_vs_opponents  # noqa: E402
from pals.bench.harness import benchmark, standard_opponents  # noqa: E402
from pals.bench.players import PALSPlayer  # noqa: E402
from pals.core.learner import run_pals  # noqa: E402
from pals.core.preference import MinimaxPreferenceOracle  # noqa: E402
from pals.envs.dots_and_boxes import (  # noqa: E402
    DotsAndBoxesEnv,
    score_margin_heuristic,
)
from pals.envs.minimax import MinimaxEnv, leftmost_leaf_heuristic  # noqa: E402
from pals.envs.nim import NimEnv, largest_pile_heuristic  # noqa: E402

SEED = 0
OUT = Path("docs/results/figures")

GAMES = [
    ("nim", "Nim (1,2,3)", NimEnv(piles=(1, 2, 3)), largest_pile_heuristic, 1),
    (
        "minimax",
        "Minimax (depth 4, b=2)",
        MinimaxEnv(depth=4, branching=2, seed=SEED),
        leftmost_leaf_heuristic,
        1,
    ),
    (
        "dots_and_boxes",
        "Dots & Boxes (1x2)",
        DotsAndBoxesEnv(rows=1, cols=2),
        score_margin_heuristic,
        2,
    ),
]


def _grouped_bar(rows, title, path: Path) -> None:
    strategies = [r.name for r in rows]
    opponents = list(rows[0].scores)
    x = np.arange(len(strategies))
    width = 0.8 / len(opponents)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, opp in enumerate(opponents):
        ax.bar(x + i * width, [r.scores[opp] for r in rows], width, label=opp)
    ax.set_xticks(x + width * (len(opponents) - 1) / 2)
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("mean reward (P2 perspective)")
    ax.set_title(f"{title} — strategies vs opponents")
    ax.legend(title="opponent")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_games() -> None:
    for slug, title, env, heuristic, oracle_depth in GAMES:
        oracle = MinimaxPreferenceOracle(env, heuristic, depth=oracle_depth)
        rows = benchmark(
            env,
            heuristic,
            oracle,
            n_games=30,
            depth_n=2,
            rollout_budget=80,
            uct_budget=150,
            qlearn_episodes=3000,
            rng=random.Random(SEED),
        )
        path = OUT / f"{slug}.png"
        _grouped_bar(rows, title, path)
        print(f"wrote {path}")


def plot_minimax_k_sweep() -> None:
    # A deeper tree with depth-N=3 leaves room for the audit to keep improving as
    # the rollout budget grows (it saturates once the reachable gains are found).
    env = MinimaxEnv(depth=8, branching=2, seed=SEED)
    heuristic = leftmost_leaf_heuristic
    oracle = MinimaxPreferenceOracle(env, heuristic, depth=1)
    opponents = standard_opponents(heuristic)
    budgets = [5, 20, 60, 150, 400]
    series: dict[str, list[float]] = {name: [] for name in opponents}

    for k in budgets:
        result = run_pals(
            env, oracle, depth_n=3, rollout_budget=k, rng=random.Random(SEED)
        )
        scores = evaluate_vs_opponents(
            env, PALSPlayer(result.model), opponents, n_games=40, rng=random.Random(1)
        )
        for name in opponents:
            series[name].append(scores[name])

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, ys in series.items():
        ax.plot(budgets, ys, marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("MCTS rollout budget K (log)")
    ax.set_ylabel("PALS mean reward")
    ax.set_title("PALS score vs rollout budget K — Minimax (depth 8, N=3)")
    ax.legend(title="opponent")
    fig.tight_layout()
    path = OUT / "minimax_k_sweep.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_games()
    plot_minimax_k_sweep()


if __name__ == "__main__":
    main()
