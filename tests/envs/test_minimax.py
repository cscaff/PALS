"""Unit tests for the Minimax game-tree environment."""

import random

import pytest

from pals.envs.base import Environment, Player
from pals.envs.minimax import MinimaxEnv, leftmost_leaf_heuristic


def test_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        MinimaxEnv(depth=0)
    with pytest.raises(ValueError):
        MinimaxEnv(depth=2, branching=0)


def test_is_an_environment():
    assert isinstance(MinimaxEnv(depth=2), Environment)


def test_structure_and_players_alternate():
    env = MinimaxEnv(depth=3, branching=2, seed=1)
    assert env.initial_state() == ()
    assert env.current_player(()) is Player.P1
    assert env.current_player((0,)) is Player.P2
    assert env.current_player((0, 1)) is Player.P1
    assert env.p1_alphabet == [0, 1]


def test_legal_actions_and_terminal():
    env = MinimaxEnv(depth=2, branching=3)
    assert env.legal_actions(()) == [0, 1, 2]
    assert not env.is_terminal((0,))
    assert env.is_terminal((0, 1))
    assert env.legal_actions((0, 1)) == []


def test_step_extends_path():
    env = MinimaxEnv(depth=2)
    assert env.step((0,), 1) == (0, 1)


def test_reward_is_leaf_value_and_reproducible():
    env_a = MinimaxEnv(depth=3, branching=2, seed=42)
    env_b = MinimaxEnv(depth=3, branching=2, seed=42)
    leaf = (0, 1, 0)
    assert env_a.reward(leaf) == env_b.reward(leaf)
    assert 0.0 <= env_a.reward(leaf) <= 100.0


def test_leftmost_heuristic_descends_first_action():
    env = MinimaxEnv(depth=3, branching=2, seed=7)
    assert leftmost_leaf_heuristic(env, ()) == env.reward((0, 0, 0))


def test_exact_oracle_matches_hand_minimax():
    # depth-2, branching-2: root is P1 (min) over P2 (max) over leaves.
    env = MinimaxEnv(depth=2, branching=2, seed=3)
    from pals.core.preference import MinimaxPreferenceOracle

    oracle = MinimaxPreferenceOracle(env, leftmost_leaf_heuristic, depth=None)
    expected = min(max(env.reward((p1, p2)) for p2 in range(2)) for p1 in range(2))
    assert oracle._value((), None) == expected


def test_run_pals_on_minimax_terminates_with_valid_machine():
    from pals.core.learner import run_pals
    from pals.core.preference import MinimaxPreferenceOracle

    env = MinimaxEnv(depth=4, branching=2, seed=5)
    oracle = MinimaxPreferenceOracle(env, leftmost_leaf_heuristic, depth=1)
    result = run_pals(env, oracle, depth_n=2, rollout_budget=30, rng=random.Random(0))
    assert len(result.model.states) >= 1
    for state in result.model.states:
        assert set(state.transitions) == set(env.p1_alphabet)
