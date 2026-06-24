"""Integration tests for the benchmark harness."""

import random

from pals.bench.harness import (
    DEFAULT_VARIANTS,
    benchmark,
    format_table,
    standard_opponents,
)
from pals.core.preference import MinimaxPreferenceOracle
from pals.envs.nim import NimEnv, largest_pile_heuristic


def _setup():
    env = NimEnv(piles=(1, 2, 3))
    oracle = MinimaxPreferenceOracle(env, largest_pile_heuristic, depth=1)
    return env, oracle


def test_standard_opponents_keys():
    opp = standard_opponents(largest_pile_heuristic)
    assert set(opp) == {"vs_random", "vs_greedy", "vs_optimal"}


def test_benchmark_produces_rows_for_variants_and_baselines():
    env, oracle = _setup()
    rows = benchmark(
        env,
        largest_pile_heuristic,
        oracle,
        n_games=4,
        depth_n=2,
        rollout_budget=10,
        uct_budget=20,
        qlearn_episodes=150,
        rng=random.Random(0),
    )
    names = {r.name for r in rows}
    assert set(DEFAULT_VARIANTS) <= names  # PALS ablation variants present
    assert {"Random", "Greedy", "Optimal", "UCT", "QLearning"} <= names
    for row in rows:
        assert set(row.scores) == {"vs_random", "vs_greedy", "vs_optimal"}
        assert all(-1.0 <= v <= 1.0 for v in row.scores.values())


def test_ablation_variants_report_their_provenance():
    env, oracle = _setup()
    rows = benchmark(
        env,
        largest_pile_heuristic,
        oracle,
        pals_variants={"PALS": {}, "PALS_no_pac": {"use_pac": False}},
        include_baselines=False,
        n_games=3,
        depth_n=2,
        rollout_budget=10,
        rng=random.Random(0),
    )
    by_name = {r.name: r for r in rows}
    # The no-PAC variant ran the MCTS audit; PALS has the standard info fields.
    assert "membership_queries" in by_name["PALS"].info
    assert by_name["PALS_no_pac"].info["accepted_deviations"] >= 0


def test_format_table_lists_strategies():
    env, oracle = _setup()
    rows = benchmark(
        env,
        largest_pile_heuristic,
        oracle,
        pals_variants={"PALS": {}},
        include_baselines=False,
        n_games=2,
        depth_n=2,
        rollout_budget=8,
        rng=random.Random(0),
    )
    table = format_table(rows)
    assert "PALS" in table
    assert "vs_random" in table
