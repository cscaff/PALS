"""Reproducible benchmark suite for the paper's results tables.

Runs the harness on each game-theoretic environment (PALS + ablations vs the
baselines, scored against random / greedy / optimal opponents), a noise-
sensitivity sweep over the preference oracle, and the active-shielding
experiment on the gas grid, printing fixed-width tables.

    python -m scripts.run_benchmarks          # all games + shielding
    python -m scripts.run_benchmarks --quick  # smaller budgets

Deterministic given the fixed seed.
"""

from __future__ import annotations

import argparse
import random

from pals.bench.evaluate import evaluate_vs_opponents, play_game
from pals.bench.harness import benchmark, format_table, standard_opponents
from pals.bench.noisy import NoisyOracle
from pals.bench.players import PALSPlayer, RandomPlayer, greedy_action
from pals.core.learner import run_pals
from pals.core.preference import MinimaxPreferenceOracle
from pals.envs.dots_and_boxes import DotsAndBoxesEnv, score_margin_heuristic
from pals.envs.gas_grid import (
    GasGridEnv,
    gas_depleted,
    manhattan_greedy_heuristic,
    safety_state_key,
)
from pals.envs.minimax import MinimaxEnv, leftmost_leaf_heuristic
from pals.envs.nim import NimEnv, largest_pile_heuristic
from pals.envs.tic_tac_toe import TicTacToeEnv, line_control_heuristic
from pals.shielding.model_check import find_violation
from pals.shielding.spec import SafetySpec

SEED = 0


def _game_rows(name, env, heuristic, *, oracle_depth, quick):
    print(f"\n### {name}")
    oracle = MinimaxPreferenceOracle(env, heuristic, depth=oracle_depth)
    rows = benchmark(
        env,
        heuristic,
        oracle,
        n_games=10 if quick else 30,
        depth_n=2,
        rollout_budget=30 if quick else 80,
        uct_budget=50 if quick else 150,
        qlearn_episodes=1000 if quick else 4000,
        rng=random.Random(SEED),
    )
    print(format_table(rows))


def run_games(quick: bool) -> None:
    _game_rows(
        "Nim (1,2,3)",
        NimEnv(piles=(1, 2, 3)),
        largest_pile_heuristic,
        oracle_depth=1,
        quick=quick,
    )
    _game_rows(
        "Minimax (depth 6, b=2)",
        MinimaxEnv(depth=6, branching=2, seed=SEED),
        leftmost_leaf_heuristic,
        oracle_depth=1,
        quick=quick,
    )
    _game_rows(
        "Tic-Tac-Toe",
        TicTacToeEnv(),
        line_control_heuristic,
        oracle_depth=2,
        quick=quick,
    )
    _game_rows(
        "Dots & Boxes (2x2)",
        DotsAndBoxesEnv(rows=2, cols=2),
        score_margin_heuristic,
        oracle_depth=2,
        quick=quick,
    )


def _noise_rows(name, env, heuristic, *, oracle_depth, quick):
    """Sweep oracle noise and report, per config, the learned automaton size,
    accepted audit deviations, and play quality vs each opponent.

    Contrasts the full PALS (audit on) against the L*+PAC core (audit off) under
    a *consistently imperfect* teacher (``NoisyOracle`` corrupts a fixed fraction
    of answers, deterministically per input). The point R1 asked about: the
    L*+PAC core stays small and robust as noise grows, whereas the MCTS audit
    accepts ever more deviations (its Thm-2 termination assumes consistent
    preferences, which noise violates) and play quality degrades.
    """
    print(f"\n### Noise sensitivity — {name}")
    print(
        f"{'config':<14}{'noise':>6}{'states':>8}{'devs':>6}"
        f"{'vs_random':>11}{'vs_greedy':>11}{'vs_optimal':>12}"
    )
    print("-" * 68)
    noises = (0.0, 0.25) if quick else (0.0, 0.1, 0.25, 0.5)
    opponents = standard_opponents(heuristic)
    n_games = 15 if quick else 30
    for use_mcts in (True, False):
        config = "PALS" if use_mcts else "PALS_no_mcts"
        for noise in noises:
            inner = MinimaxPreferenceOracle(env, heuristic, depth=oracle_depth)
            oracle = NoisyOracle(inner, env, noise=noise, seed=SEED)
            result = run_pals(
                env,
                oracle,
                use_mcts=use_mcts,
                use_pac=True,
                depth_n=2,
                rollout_budget=30 if quick else 80,
                rng=random.Random(SEED),
            )
            scores = evaluate_vs_opponents(
                env,
                PALSPlayer(result.model),
                opponents,
                n_games=n_games,
                rng=random.Random(SEED),
            )
            print(
                f"{config:<14}{noise:>6.2f}{len(result.model.states):>8}"
                f"{result.accepted_deviations:>6}"
                f"{scores['vs_random']:>11.3f}{scores['vs_greedy']:>11.3f}"
                f"{scores['vs_optimal']:>12.3f}"
            )


def run_noise(quick: bool) -> None:
    # Fast games only: under high noise the audit's deviation count (and so the
    # automaton, and runtime) blows up on large-alphabet games like Tic-Tac-Toe
    # — that blowup is documented in docs/03_paper_alignment.md §4 rather than
    # baked into the default suite.
    _noise_rows(
        "Nim (1,2,3)",
        NimEnv(piles=(1, 2, 3)),
        largest_pile_heuristic,
        oracle_depth=1,
        quick=quick,
    )
    _noise_rows(
        "Minimax (depth 6, b=2)",
        MinimaxEnv(depth=6, branching=2, seed=SEED),
        leftmost_leaf_heuristic,
        oracle_depth=1,
        quick=quick,
    )


def run_shielding() -> None:
    print("\n### Active shielding — gas corridor, spec G(gas>0)")
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
    unsafe = find_violation(env, unshielded.model, gas_depleted)
    safe = find_violation(env, shielded.model, gas_depleted)
    delivers = play_game(
        env, RandomPlayer(), PALSPlayer(shielded.model), random.Random(SEED)
    )
    print(f"  unshielded: reachable G(gas>0) violation = {unsafe is not None}")
    print(
        f"  shielded:   reachable G(gas>0) violation = {safe is not None}"
        f"  (safe patches = {shielded.shield_patches}, delivers reward = {delivers})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_games(args.quick)
    run_noise(args.quick)
    run_shielding()


if __name__ == "__main__":
    main()
