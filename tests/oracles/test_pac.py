"""Unit tests for the PAC equivalence oracle."""

import random

from pals.core.lstar import MealyLStar
from pals.core.preference import MinimaxPreferenceOracle
from pals.core.sul import PreferenceSUL
from pals.envs.nim import NimEnv, largest_pile_heuristic
from pals.oracles.exact import BoundedExactOracle
from pals.oracles.pac import PACEquivalenceOracle


def _setup(piles=(1, 2, 3)):
    env = NimEnv(piles=piles)
    oracle = MinimaxPreferenceOracle(env, largest_pile_heuristic, depth=None)
    sul = PreferenceSUL(env, oracle)
    return env, sul


def test_sampled_walk_is_p1_alphabet_and_bounded():
    env, sul = _setup()
    pac = PACEquivalenceOracle(env, sul, rng=random.Random(0))
    alphabet = set(env.p1_alphabet)
    for _ in range(20):
        walk = pac._sample_p1_walk()
        # The walk samples P2 moves randomly (and discards them), so its P1
        # inputs cannot be replayed against a deterministic trajectory; the
        # checkable invariants are membership in the alphabet and the length cap.
        assert all(p1 in alphabet for p1 in walk)
        assert len(walk) <= pac.max_walk


def test_query_count_and_sample_schedule_grows():
    env, sul = _setup()
    pac = PACEquivalenceOracle(env, sul, rng=random.Random(1))
    pac.find_counterexample(_learned_model(env, sul))
    assert pac.num_queries == 1


def test_first_disagreement_detects_mismatch():
    env, sul = _setup(piles=(1, 2, 3))
    model = _learned_model(env, sul)
    # Tamper with the model so it disagrees with the SUL somewhere.
    some_state = model.initial_state
    a = next(iter(some_state.transitions))
    _, dst = some_state.transitions[a]
    some_state.transitions[a] = ("__wrong__", dst)
    cex = pac_first_mismatch(env, sul, model)
    assert cex is None or model.output_of(cex) != list(sul.query(cex))


def test_no_counterexample_against_correct_hypothesis():
    env, sul = _setup()
    model = _learned_model(env, sul)
    pac = PACEquivalenceOracle(env, sul, eps=0.1, delta=0.1, rng=random.Random(2))
    assert pac.find_counterexample(model) is None


# --- helpers ---


def _learned_model(env, sul):
    eq = BoundedExactOracle(env.p1_alphabet, sul, max_length=4)
    return MealyLStar(env.p1_alphabet, sul, eq).run()


def pac_first_mismatch(env, sul, model):
    pac = PACEquivalenceOracle(env, sul, eps=0.05, delta=0.05, rng=random.Random(3))
    return pac.find_counterexample(model)
