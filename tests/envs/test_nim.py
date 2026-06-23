"""Unit tests for the Nim environment."""

import pytest

from pals.envs.base import Player
from pals.envs.nim import NimEnv, NimState


def test_rejects_negative_piles():
    with pytest.raises(ValueError):
        NimEnv(piles=(1, -2, 3))


def test_initial_state():
    env = NimEnv(piles=(1, 2, 3))
    s = env.initial_state()
    assert s == NimState(piles=(1, 2, 3), player=Player.P1)
    assert env.current_player(s) is Player.P1
    assert not env.is_terminal(s)


def test_legal_actions_enumerates_pile_and_count():
    env = NimEnv(piles=(1, 2, 3))
    actions = env.legal_actions(env.initial_state())
    expected = {(0, 1), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3)}
    assert set(actions) == expected
    assert len(actions) == len(expected)  # no duplicates


def test_p1_alphabet_is_root_actions():
    env = NimEnv(piles=(1, 2, 3))
    assert set(env.p1_alphabet) == set(env.legal_actions(env.initial_state()))


def test_step_removes_objects_and_flips_player():
    env = NimEnv(piles=(1, 2, 3))
    s = env.step(env.initial_state(), (2, 3))
    assert s == NimState(piles=(1, 2, 0), player=Player.P2)


def test_step_does_not_mutate_input_state():
    env = NimEnv(piles=(1, 2, 3))
    s0 = env.initial_state()
    env.step(s0, (2, 1))
    assert s0.piles == (1, 2, 3)  # tuples are immutable; defensive check


def test_terminal_detection():
    env = NimEnv()
    assert env.is_terminal(NimState(piles=(0, 0, 0), player=Player.P1))
    assert not env.is_terminal(NimState(piles=(0, 1, 0), player=Player.P2))


def test_legal_actions_empty_at_terminal():
    env = NimEnv()
    assert env.legal_actions(NimState(piles=(0, 0), player=Player.P2)) == []


def test_reward_from_p2_perspective():
    env = NimEnv()
    # Terminal with P2 to move => P1 took the last object => P1 won => -1.
    assert env.reward(NimState(piles=(0,), player=Player.P2)) == -1.0
    # Terminal with P1 to move => P2 took the last object => P2 won => +1.
    assert env.reward(NimState(piles=(0,), player=Player.P1)) == 1.0


def test_single_pile_full_game():
    env = NimEnv(piles=(1,))
    terminal = env.get_node([(0, 1)])
    assert terminal == NimState(piles=(0,), player=Player.P2)
    assert env.is_terminal(terminal)
    # P1 took the last object and wins; reward from P2's view is -1.
    assert env.reward(terminal) == -1.0
