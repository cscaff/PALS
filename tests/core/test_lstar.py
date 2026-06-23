"""Unit and integration tests for Mealy L*.

With an *exact* (global) preference oracle, PALS reduces to standard L*, so a
bounded-exact equivalence oracle must recover a hypothesis equivalent to the SUL
over all bounded inputs.
"""

import itertools

from pals.core.lstar import MealyLStar, MealyMachine, MealyState
from pals.core.preference import MinimaxPreferenceOracle
from pals.core.sul import PreferenceSUL
from pals.envs.nim import NimEnv, largest_pile_heuristic
from pals.oracles.exact import BoundedExactOracle


def _learn(piles, max_length):
    env = NimEnv(piles=piles)
    oracle = MinimaxPreferenceOracle(env, largest_pile_heuristic, depth=None)
    sul = PreferenceSUL(env, oracle)
    eq = BoundedExactOracle(env.p1_alphabet, sul, max_length=max_length)
    model = MealyLStar(env.p1_alphabet, sul, eq).run()
    return env, sul, model


def test_mealy_machine_step_and_output():
    a = MealyState("q0")
    b = MealyState("q1")
    a.transitions["x"] = ("o", b)
    m = MealyMachine([a, b], a)
    assert m.output_of(["x"]) == ["o"]
    assert m.output_of(["nope"]) == [None]  # no edge -> sink


def test_learns_single_pile():
    env, sul, model = _learn(piles=(1,), max_length=2)
    assert len(model.states) >= 1
    assert model.output_of([(0, 1)]) == list(sul.query([(0, 1)]))


def test_learned_machine_matches_sul_over_bounded_inputs():
    env, sul, model = _learn(piles=(1, 2, 3), max_length=4)
    for length in range(1, 5):
        for seq in itertools.product(env.p1_alphabet, repeat=length):
            assert model.output_of(seq) == list(sul.query(seq)), seq


def test_converges_with_no_residual_counterexample():
    env, sul, model = _learn(piles=(1, 2, 3), max_length=4)
    eq = BoundedExactOracle(env.p1_alphabet, sul, max_length=4)
    assert eq.find_counterexample(model) is None
