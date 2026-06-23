"""Unit tests for the (strictly-alternating) Dots & Boxes environment."""

import random

from pals.envs.base import Environment, Player
from pals.envs.dots_and_boxes import (
    DotsAndBoxesEnv,
    score_margin_heuristic,
)


def test_rejects_bad_dimensions():
    import pytest

    with pytest.raises(ValueError):
        DotsAndBoxesEnv(rows=0, cols=2)


def test_is_an_environment():
    assert isinstance(DotsAndBoxesEnv(), Environment)


def test_edge_and_box_counts():
    env = DotsAndBoxesEnv(rows=1, cols=2)
    # horizontal (1+1)*2 = 4, vertical 1*(2+1) = 3 -> 7 edges, 2 boxes.
    assert env.n_edges == 7
    assert env.n_boxes == 2
    assert env.p1_alphabet == list(range(7))


def test_initial_state():
    env = DotsAndBoxesEnv()
    s = env.initial_state()
    assert s.drawn == frozenset()
    assert s.player is Player.P1
    assert len(env.legal_actions(s)) == env.n_edges


def test_completing_a_box_scores_a_point():
    # 1x1 board: 4 edges (0,1 horizontal; 2,3 vertical), a single box.
    env = DotsAndBoxesEnv(rows=1, cols=1)
    assert env.n_edges == 4 and env.n_boxes == 1
    s = env.initial_state()
    # Draw three edges (no completion yet), then the fourth completes the box.
    for e in (0, 1, 2):
        s = env.step(s, e)
    assert s.p1_score == 0 and s.p2_score == 0
    mover = s.player
    s = env.step(s, 3)
    # Whoever drew the 4th edge scores the box.
    scored = s.p1_score if mover is Player.P1 else s.p2_score
    assert scored == 1
    assert env.is_terminal(s)


def test_legal_actions_shrink_and_terminal():
    env = DotsAndBoxesEnv(rows=1, cols=1)
    s = env.step(env.initial_state(), 0)
    assert 0 not in env.legal_actions(s)
    assert len(env.legal_actions(s)) == env.n_edges - 1


def test_reward_perspective():
    env = DotsAndBoxesEnv(rows=1, cols=2)
    from pals.envs.dots_and_boxes import DBState

    assert env.reward(DBState(frozenset(), 0, 2, Player.P1)) == 1.0  # P2 ahead
    assert env.reward(DBState(frozenset(), 2, 0, Player.P1)) == -1.0
    assert env.reward(DBState(frozenset(), 1, 1, Player.P1)) == 0.0


def test_heuristic_sign():
    env = DotsAndBoxesEnv(rows=1, cols=2)
    from pals.envs.dots_and_boxes import DBState

    ahead = DBState(frozenset(), 0, 1, Player.P1)
    behind = DBState(frozenset(), 1, 0, Player.P1)
    assert score_margin_heuristic(env, ahead) > 0
    assert score_margin_heuristic(env, behind) < 0


def test_run_pals_terminates_with_valid_machine():
    from pals.core.learner import run_pals
    from pals.core.preference import MinimaxPreferenceOracle

    env = DotsAndBoxesEnv(rows=1, cols=1)
    oracle = MinimaxPreferenceOracle(env, score_margin_heuristic, depth=2)
    result = run_pals(env, oracle, depth_n=2, rollout_budget=20, rng=random.Random(0))
    assert len(result.model.states) >= 1
    for state in result.model.states:
        assert set(state.transitions) == set(env.p1_alphabet)
