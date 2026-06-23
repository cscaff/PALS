"""Unit tests for the MCTS audit oracle."""

import random

from pals.core.lstar import MealyLStar
from pals.core.preference import MinimaxPreferenceOracle
from pals.core.sul import PreferenceSUL
from pals.core.table_b import TableB
from pals.envs.base import Player
from pals.envs.nim import NimEnv, largest_pile_heuristic
from pals.oracles.exact import BoundedExactOracle
from pals.oracles.mcts_audit import MCTSAuditOracle


def _audit(piles, depth, depth_n=3, budget=50, seed=0):
    env = NimEnv(piles=piles)
    oracle = MinimaxPreferenceOracle(env, largest_pile_heuristic, depth=depth)
    sul = PreferenceSUL(env, oracle)
    table_b = TableB()
    audit = MCTSAuditOracle(
        env,
        sul,
        oracle,
        table_b,
        depth_n=depth_n,
        rollout_budget=budget,
        rng=random.Random(seed),
    )
    return env, oracle, sul, audit


def _learn_with_exact(env, sul):
    eq = BoundedExactOracle(env.p1_alphabet, sul, max_length=6)
    return MealyLStar(env.p1_alphabet, sul, eq).run()


def test_no_deviation_when_p2_has_no_choice():
    # Nim (2,): P2's only decision node has a single legal move, so no
    # alternative deviation exists -> the audit cannot fire.
    env, oracle, sul, audit = _audit(piles=(2,), depth=None)
    model = _learn_with_exact(env, sul)
    assert audit.find_counterexample(model) is None


def test_sample_deviation_point_is_a_p2_turn_prefix():
    env, oracle, sul, audit = _audit(piles=(1, 2, 3), depth=None)
    audit.hypothesis = _learn_with_exact(env, sul)
    for _ in range(20):
        sub = audit._sample_deviation_point()
        if sub is None:
            continue
        # Prefix ends at a P2 turn and has odd length (ends on a P1 input).
        assert len(sub) % 2 == 1
        assert env.current_player_at(sub) is Player.P2


def test_choose_deviation_differs_from_hypothesis():
    env, oracle, sul, audit = _audit(piles=(1, 2, 3), depth=None)
    audit.hypothesis = _learn_with_exact(env, sul)
    sub = [(0, 1)]  # after P1 empties pile 0, P2 to move with several options
    lam_h = audit._hyp_output(sul.p1_inputs_of(sub))
    dev = audit._choose_deviation(sub)
    assert dev is not None
    assert dev != lam_h
    assert dev in env.p2_legal_moves(sub)


def test_vote_prefers_winning_deviation():
    # On Nim (2,): leaving a single object for P2 wins; emptying loses for P2.
    env, oracle, sul, audit = _audit(piles=(2,), depth=None)
    audit.hypothesis = None  # _vote_and_value doesn't need the hypothesis
    dev_leaves = [[(0, 1)]]  # P1 took 1 -> P2 faces (1,) and will win  (+1)
    hyp_leaves = [[(0, 2)]]  # P1 took 2 -> game over, P1 took last      (-1)
    _, dev_preferred = audit._vote_and_value(dev_leaves, hyp_leaves)
    assert dev_preferred is True
    _, reversed_pref = audit._vote_and_value(hyp_leaves, dev_leaves)
    assert reversed_pref is False


def test_converged_flag_latches():
    env, oracle, sul, audit = _audit(piles=(2,), depth=None)
    model = _learn_with_exact(env, sul)
    assert audit.find_counterexample(model) is None
    assert audit._converged is True
    # Once converged, further calls are immediate no-ops.
    assert audit.find_counterexample(model) is None


def test_accepted_cex_is_valid_and_mutates_sul():
    # Suboptimal oracle leaves room for the audit to improve; if it fires, the
    # returned CEX must genuinely separate the hypothesis from the (now-mutated)
    # SUL.
    env, oracle, sul, audit = _audit(piles=(1, 2, 3), depth=1, budget=200, seed=1)
    eq = BoundedExactOracle(env.p1_alphabet, sul, max_length=6)
    model = MealyLStar(env.p1_alphabet, sul, eq).run()
    cex = audit.find_counterexample(model)
    if cex is not None:
        assert all(a in env.p1_alphabet for a in cex)
        assert model.output_of(cex) != list(sul.query(cex))
        assert audit.accepted_deviations >= 1
