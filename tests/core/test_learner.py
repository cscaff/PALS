"""End-to-end tests for the full PALS learning loop (run_pals)."""

import itertools
import random

import pytest

from pals.core.learner import run_pals
from pals.core.preference import MinimaxPreferenceOracle
from pals.envs.nim import NimEnv, largest_pile_heuristic


def _oracle(env, depth):
    return MinimaxPreferenceOracle(env, largest_pile_heuristic, depth=depth)


def test_requires_a_stage():
    env = NimEnv(piles=(1, 2, 3))
    with pytest.raises(ValueError):
        run_pals(env, _oracle(env, None), use_mcts=False, use_pac=False)


def test_exact_oracle_learns_sul_equivalent_machine():
    env = NimEnv(piles=(1, 2, 3))
    result = run_pals(
        env,
        _oracle(env, None),
        depth_n=3,
        rollout_budget=50,
        rng=random.Random(0),
    )
    # Whatever deviations fired, the final machine must match the final SUL.
    for length in range(1, 5):
        for seq in itertools.product(env.p1_alphabet, repeat=length):
            assert result.model.output_of(seq) == list(result.sul.query(seq)), seq


def test_runs_with_suboptimal_oracle_and_is_valid():
    env = NimEnv(piles=(1, 2, 3))
    result = run_pals(
        env,
        _oracle(env, 1),
        depth_n=3,
        rollout_budget=100,
        rng=random.Random(0),
    )
    assert len(result.model.states) >= 1
    assert result.membership_queries > 0
    # Deterministic, total Mealy machine: every state has every input wired.
    for state in result.model.states:
        assert set(state.transitions) == set(env.p1_alphabet)


def test_ablation_pure_lstar_pac_only():
    env = NimEnv(piles=(1, 2, 3))
    result = run_pals(
        env,
        _oracle(env, None),
        use_mcts=False,
        use_pac=True,
        rng=random.Random(0),
    )
    assert result.accepted_deviations == 0  # no audit ran
    assert len(result.model.states) >= 1


def test_ablation_mcts_only():
    env = NimEnv(piles=(1, 2, 3))
    result = run_pals(
        env,
        _oracle(env, 1),
        use_mcts=True,
        use_pac=False,
        depth_n=3,
        rollout_budget=50,
        rng=random.Random(0),
    )
    assert len(result.model.states) >= 1


def test_audit_fires_on_some_seed_with_suboptimal_oracle():
    # The audit must actually do something: across seeds, a suboptimal oracle
    # should yield at least one accepted improving deviation.
    total = 0
    for seed in range(5):
        env = NimEnv(piles=(1, 2, 3))
        result = run_pals(
            env,
            _oracle(env, 1),
            depth_n=3,
            rollout_budget=150,
            use_pac=False,
            rng=random.Random(seed),
        )
        total += result.accepted_deviations
    assert total > 0
