"""Unit tests for the preference SUL (membership queries + mutation)."""

from pals.core.preference import MinimaxPreferenceOracle
from pals.core.sul import PreferenceSUL
from pals.envs.nim import NimEnv, largest_pile_heuristic


def _sul(piles=(1, 2, 3), depth=None):
    env = NimEnv(piles=piles)
    oracle = MinimaxPreferenceOracle(env, largest_pile_heuristic, depth=depth)
    return env, oracle, PreferenceSUL(env, oracle)


def test_query_terminal_after_p1_is_sink():
    env, _, sul = _sul(piles=(1,))
    # P1 takes the only object -> terminal -> P2 has no response.
    assert sul.query([(0, 1)]) == (None,)


def test_query_returns_legal_p2_response():
    env, _, sul = _sul()
    out = sul.query([(0, 1)])
    assert out[0] in env.p2_legal_moves([(0, 1)])


def test_query_illegal_p1_is_sink():
    _, _, sul = _sul()
    assert sul.query([(9, 9)]) == (None,)


def test_query_sink_persists_after_off_tree():
    _, _, sul = _sul(piles=(1,))
    # First input reaches terminal; everything after is a sink.
    assert sul.query([(0, 1), (0, 1), (0, 1)]) == (None, None, None)


def test_query_is_deterministic():
    _, _, sul = _sul()
    assert sul.query([(2, 1), (1, 1)]) == sul.query([(2, 1), (1, 1)])


def test_query_increments_counter():
    _, _, sul = _sul()
    before = sul.num_queries
    sul.query([(0, 1)])
    sul.query([(0, 1)])
    assert sul.num_queries == before + 2


def test_update_strategy_changes_response():
    env, oracle, sul = _sul()
    default = oracle.preferred_move([(0, 1)])
    alternative = next(m for m in env.p2_legal_moves([(0, 1)]) if m != default)
    assert sul.update_strategy([(0, 1)], alternative) is True
    assert sul.query([(0, 1)])[0] == alternative


def test_patch_locks_against_update():
    env, oracle, sul = _sul()
    legal = env.p2_legal_moves([(0, 1)])
    sul.patch([(0, 1)], legal[0])
    # A later strategy update at a locked site is refused.
    assert sul.update_strategy([(0, 1)], legal[1]) is False
    assert sul.query([(0, 1)])[0] == legal[0]


def test_current_response_none_off_p2_turn():
    _, _, sul = _sul()
    assert sul.current_response([]) is None  # root is P1's turn


def test_current_response_matches_query():
    _, _, sul = _sul()
    assert sul.current_response([(0, 1)]) == sul.query([(0, 1)])[0]


def test_p1_inputs_of_takes_even_indices():
    assert PreferenceSUL.p1_inputs_of(["a", "b", "c", "d"]) == ["a", "c"]
