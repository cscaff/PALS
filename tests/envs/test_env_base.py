"""Unit tests for the generic trace-navigation helpers on ``Environment``.

These exercise the derived methods (``get_node`` and friends) through a
concrete environment (Nim), since they are defined in terms of the abstract
primitives.
"""

from pals.envs.base import Environment, Player
from pals.envs.nim import NimEnv, NimState


def test_get_node_empty_trace_is_initial():
    env = NimEnv(piles=(1, 2, 3))
    assert env.get_node([]) == env.initial_state()


def test_get_node_follows_trace():
    env = NimEnv(piles=(1, 2, 3))
    # P1 empties pile 0, P2 takes one from pile 1.
    node = env.get_node([(0, 1), (1, 1)])
    assert node == NimState(piles=(0, 1, 3), player=Player.P1)


def test_get_node_illegal_action_returns_none():
    env = NimEnv(piles=(1, 2, 3))
    assert env.get_node([(0, 9)]) is None  # can't take 9 from a pile of 1


def test_get_node_past_terminal_returns_none():
    env = NimEnv(piles=(1,))
    # First action reaches terminal; a second action runs off the tree.
    assert env.get_node([(0, 1), (0, 1)]) is None


def test_current_player_at_alternates():
    env = NimEnv(piles=(1, 2, 3))
    assert env.current_player_at([]) is Player.P1
    assert env.current_player_at([(0, 1)]) is Player.P2
    assert env.current_player_at([(0, 1), (1, 1)]) is Player.P1


def test_current_player_at_terminal_is_none():
    env = NimEnv(piles=(1,))
    assert env.current_player_at([(0, 1)]) is None


def test_current_player_at_invalid_trace_is_none():
    env = NimEnv(piles=(1, 2, 3))
    assert env.current_player_at([(0, 9)]) is None


def test_p1_legal_inputs_only_on_p1_turn():
    env = NimEnv(piles=(1, 2, 3))
    assert set(env.p1_legal_inputs([])) == set(env.legal_actions(env.initial_state()))
    assert env.p1_legal_inputs([(0, 1)]) == []  # P2's turn


def test_p2_legal_moves_only_on_p2_turn():
    env = NimEnv(piles=(1, 2, 3))
    assert env.p2_legal_moves([]) == []  # P1's turn at the root
    after_p1 = env.p2_legal_moves([(0, 1)])
    assert set(after_p1) == {(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)}


def test_legal_actions_at_terminal_is_empty():
    env = NimEnv(piles=(1,))
    assert env.legal_actions_at([(0, 1)]) == []


def test_nim_is_an_environment():
    assert isinstance(NimEnv(), Environment)
