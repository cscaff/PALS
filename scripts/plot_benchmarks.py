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
from pals.bench.players import PALSPlayer, RandomPlayer, greedy_action  # noqa: E402
from pals.core.learner import run_pals  # noqa: E402
from pals.core.preference import MinimaxPreferenceOracle  # noqa: E402
from pals.envs.base import Player  # noqa: E402
from pals.envs.dots_and_boxes import (  # noqa: E402
    DotsAndBoxesEnv,
    score_margin_heuristic,
)
from pals.envs.gas_grid import (  # noqa: E402
    GasGridEnv,
    gas_depleted,
    manhattan_greedy_heuristic,
    safety_state_key,
)
from pals.envs.minimax import MinimaxEnv, leftmost_leaf_heuristic  # noqa: E402
from pals.envs.nim import NimEnv, largest_pile_heuristic  # noqa: E402
from pals.envs.tic_tac_toe import (  # noqa: E402
    TicTacToeEnv,
    line_control_heuristic,
)
from pals.shielding.spec import SafetySpec  # noqa: E402

SEED = 0
OUT = Path("docs/results/figures")

GAMES = [
    ("nim", "Nim (1,2,3)", NimEnv(piles=(1, 2, 3)), largest_pile_heuristic, 1),
    (
        "minimax",
        "Minimax (depth 6, b=2)",
        MinimaxEnv(depth=6, branching=2, seed=SEED),
        leftmost_leaf_heuristic,
        1,
    ),
    (
        "tic_tac_toe",
        "Tic-Tac-Toe",
        TicTacToeEnv(),
        line_control_heuristic,
        2,
    ),
    (
        "dots_and_boxes",
        "Dots & Boxes (2x2)",
        DotsAndBoxesEnv(rows=2, cols=2),
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


def _gas_trajectory(env, model) -> list[int]:
    """Replay the learned controller on the corridor; return gas after each P2
    action (P1 turns are forced reveals/task and don't change gas)."""
    p1, p2 = RandomPlayer(), PALSPlayer(model)
    rng = random.Random(SEED)
    trace: list = []
    state = env.initial_state()
    gas = [state.gas]
    while not env.is_terminal(state):
        is_p2 = env.current_player(state) is Player.P2
        action = (p2 if is_p2 else p1).action(env, trace, rng)
        trace.append(action)
        state = env.step(state, action)
        if is_p2:
            gas.append(state.gas)
    return gas


def plot_shielding() -> None:
    env = GasGridEnv(
        rows=1,
        cols=3,
        home=(0, 0),
        refuel=(0, 1),
        dropoff=(0, 2),
        gas_max=2,
        eligible_cells=((0, 0),),
        gas_is_fatal=False,
    )
    oracle = MinimaxPreferenceOracle(env, manhattan_greedy_heuristic, depth=4)
    spec = SafetySpec(gas_depleted, name="G(gas>0)", state_key=safety_state_key)

    unshielded = run_pals(
        env,
        oracle,
        depth_n=2,
        rollout_budget=10,
        use_pac=False,
        rng=random.Random(SEED),
    )
    shielded = run_pals(
        env,
        oracle,
        depth_n=2,
        rollout_budget=10,
        use_pac=False,
        spec=spec,
        prefer_action=greedy_action(env, manhattan_greedy_heuristic),
        rng=random.Random(SEED),
    )
    g_unsafe = _gas_trajectory(env, unshielded.model)
    g_safe = _gas_trajectory(env, shielded.model)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        range(len(g_unsafe)),
        g_unsafe,
        marker="o",
        color="tab:red",
        label="unshielded (gas-blind preference)",
    )
    ax.plot(
        range(len(g_safe)),
        g_safe,
        marker="s",
        color="tab:green",
        label="shielded (refuels to stay safe)",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    if 0 in g_unsafe:
        i = g_unsafe.index(0)
        ax.annotate(
            "G(gas>0) violated",
            xy=(i, 0),
            xytext=(i - 1.4, 0.6),
            arrowprops={"arrowstyle": "->"},
            color="tab:red",
        )
    ax.set_xlabel("controller step")
    ax.set_ylabel("gas")
    ax.set_yticks(range(-1, env.gas_max + 1))
    ax.set_title("Active shielding on the gas corridor — spec G(gas > 0)")
    ax.legend()
    fig.tight_layout()
    path = OUT / "shielding.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_games()
    plot_minimax_k_sweep()
    plot_shielding()


if __name__ == "__main__":
    main()
