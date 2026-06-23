"""Unit tests for the SMT preference valuer."""

from pals.core.preference import Preference
from pals.core.smt import SMTValuer


def test_empty_solve_returns_empty():
    assert SMTValuer().solve() == {}


def test_strict_preference_orders_values():
    s = SMTValuer()
    s.add(["a"], ["b"], Preference.FIRST)
    values = s.solve()
    assert values is not None
    assert values[("a",)] > values[("b",)]


def test_values_normalized_to_unit_interval():
    s = SMTValuer()
    s.add(["a"], ["b"], Preference.FIRST)
    s.add(["b"], ["c"], Preference.FIRST)
    values = s.solve()
    assert min(values.values()) == 0.0
    assert max(values.values()) == 1.0
    assert values[("a",)] > values[("b",)] > values[("c",)]


def test_equal_preference_ties_values():
    s = SMTValuer()
    s.add(["a"], ["b"], Preference.EQUAL)
    values = s.solve()
    assert values[("a",)] == values[("b",)]


def test_all_equal_collapses_to_half():
    s = SMTValuer()
    s.add(["a"], ["b"], Preference.EQUAL)
    values = s.solve()
    # hi == lo branch: every value defaults to 0.5.
    assert values[("a",)] == 0.5
    assert values[("b",)] == 0.5


def test_inconsistent_constraints_unsat():
    s = SMTValuer()
    s.add(["a"], ["b"], Preference.FIRST)
    s.add(["b"], ["a"], Preference.FIRST)  # contradicts the first
    assert s.solve() is None
    assert not s.is_satisfiable()


def test_value_lookup_after_solve():
    s = SMTValuer()
    s.add(["x"], ["y"], Preference.SECOND)  # y preferred
    s.solve()
    assert s.value(["y"]) > s.value(["x"])
    assert s.value(["unseen"]) is None
