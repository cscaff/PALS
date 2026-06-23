"""Unit tests for the composite equivalence oracle."""

import pytest

from pals.oracles.composite import CompositeEquivalenceOracle


class _Stage:
    def __init__(self, cex):
        self.cex = cex
        self.calls = 0

    def find_counterexample(self, hypothesis):
        self.calls += 1
        return self.cex


def test_requires_at_least_one_stage():
    with pytest.raises(ValueError):
        CompositeEquivalenceOracle()


def test_returns_first_non_none_and_short_circuits():
    first = _Stage(None)
    second = _Stage(["a"])
    third = _Stage(["b"])
    composite = CompositeEquivalenceOracle(first, second, third)
    assert composite.find_counterexample(None) == ["a"]
    assert first.calls == 1
    assert second.calls == 1
    assert third.calls == 0  # short-circuited


def test_returns_none_when_all_stages_agree():
    composite = CompositeEquivalenceOracle(_Stage(None), _Stage(None))
    assert composite.find_counterexample(None) is None


def test_counts_queries():
    composite = CompositeEquivalenceOracle(_Stage(None))
    composite.find_counterexample(None)
    composite.find_counterexample(None)
    assert composite.num_queries == 2
