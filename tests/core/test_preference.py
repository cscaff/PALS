"""Unit tests for the preference oracle interface and minimax implementation."""

from pals.core.preference import (
    MinimaxPreferenceOracle,
    Preference,
    PreferenceOracle,
)
from pals.envs.nim import NimEnv, largest_pile_heuristic


def _exact_oracle(piles=(1, 2, 3)):
    env = NimEnv(piles=piles)
    return env, MinimaxPreferenceOracle(env, largest_pile_heuristic, depth=None)


def test_satisfies_protocol():
    _, oracle = _exact_oracle()
    assert isinstance(oracle, PreferenceOracle)


def test_terminal_value_is_reward():
    env, oracle = _exact_oracle(piles=(1,))
    terminal = env.get_node([(0, 1)])
    assert oracle._value(terminal, oracle.depth) == env.reward(terminal)


def test_preferred_move_none_off_p2_turn():
    env, oracle = _exact_oracle()
    # Root is P1's turn → no P2 preferred move.
    assert oracle.preferred_move([]) is None
    # Terminal → no move.
    env1, oracle1 = _exact_oracle(piles=(1,))
    assert oracle1.preferred_move([(0, 1)]) is None


def test_preferred_move_is_legal_p2_action():
    env, oracle = _exact_oracle()
    move = oracle.preferred_move([(0, 1)])  # now P2's turn
    assert move in env.p2_legal_moves([(0, 1)])


def test_compare_orders_by_value():
    env, oracle = _exact_oracle(piles=(2,))
    # From (2,), P1 can take 1 (-> P2 faces (1,)) or 2 (-> P2 faces (0,), P2 lost).
    win_for_p2 = [(0, 1)]  # leaves (1,) for P2, who takes it and wins → reward +1
    loss_for_p2 = [(0, 2)]  # empties the pile; P1 took last → P2 lost → reward -1
    assert oracle.compare(win_for_p2, loss_for_p2) is Preference.FIRST
    assert oracle.compare(loss_for_p2, win_for_p2) is Preference.SECOND


def test_compare_equal_traces():
    _, oracle = _exact_oracle()
    assert oracle.compare([(0, 1)], [(0, 1)]) is Preference.EQUAL


def test_exact_oracle_plays_optimally_on_tiny_nim():
    # Nim (1,): P1 must take the last object and win; P2 cannot win.
    env, oracle = _exact_oracle(piles=(1,))
    root_value = oracle._value(env.initial_state(), None)
    assert root_value == -1.0  # P1 wins from the start under optimal play


def test_depth_limit_uses_heuristic():
    env = NimEnv(piles=(1, 2, 3))
    shallow = MinimaxPreferenceOracle(env, largest_pile_heuristic, depth=0)
    # At depth 0 on a non-terminal state, the value is exactly the heuristic.
    state = env.initial_state()
    assert shallow._value(state, 0) == largest_pile_heuristic(env, state)


def test_caching_returns_consistent_values():
    env, oracle = _exact_oracle()
    state = env.initial_state()
    first = oracle._value(state, oracle.depth)
    second = oracle._value(state, oracle.depth)
    assert first == second
    assert (state, oracle.depth) in oracle._cache
