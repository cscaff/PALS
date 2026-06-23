"""Unit tests for Table B (the MCTS exploration record)."""

import random

from pals.core.table_b import UNVISITED_SCORE, TableB


def test_unseen_node_has_no_actions():
    assert TableB().actions_at(["a"]) == {}


def test_record_visit_increments():
    t = TableB()
    t.record_visit(["x"], "a")
    t.record_visit(["x"], "a")
    assert t.actions_at(["x"])["a"].visits == 2


def test_pruned_action_not_visited_and_scores_neg_inf():
    t = TableB()
    t.prune(["x"], "a")
    t.record_visit(["x"], "a")  # ignored while pruned
    assert t.actions_at(["x"])["a"].visits == 0
    assert t.ucb_score(["x"], "a", ["a", "b"]) == float("-inf")


def test_unvisited_action_gets_high_prior():
    t = TableB()
    assert t.ucb_score(["x"], "a", ["a", "b"]) == UNVISITED_SCORE


def test_ucb_combines_value_and_bonus():
    t = TableB()
    # Visit both so totals are positive and the bonus is finite.
    t.record_visit(["x"], "a")
    t.record_visit(["x"], "b")
    t.update_value(["x"], "a", 0.9)
    score = t.ucb_score(["x"], "a", ["a", "b"])
    assert score > 0.9  # value plus a positive exploration bonus


def test_best_action_picks_highest_score():
    t = TableB()
    t.record_visit(["x"], "a")
    t.record_visit(["x"], "b")
    t.update_value(["x"], "a", 0.1)
    t.update_value(["x"], "b", 0.9)
    assert t.best_action(["x"], ["a", "b"]) == "b"


def test_best_action_none_when_all_pruned():
    t = TableB()
    t.prune(["x"], "a")
    assert t.best_action(["x"], ["a"]) is None


def test_sample_action_respects_pruning():
    t = TableB()
    t.prune(["x"], "a")
    rng = random.Random(0)
    # Only "b" is live, so it must always be sampled.
    assert all(t.sample_action(["x"], ["a", "b"], rng=rng) == "b" for _ in range(20))


def test_sample_action_none_when_empty():
    assert TableB().sample_action(["x"], []) is None


def test_summary_counts():
    t = TableB()
    t.record_visit(["x"], "a")
    t.record_visit(["x"], "b")
    t.prune(["x"], "b")
    summary = t.summary()
    assert "2 edges" in summary
    assert "1 pruned" in summary
