"""Unit tests for the bounded exact equivalence oracle."""

from pals.core.lstar import MealyMachine, MealyState
from pals.core.preference import MinimaxPreferenceOracle
from pals.core.sul import PreferenceSUL
from pals.envs.nim import NimEnv, largest_pile_heuristic
from pals.oracles.exact import BoundedExactOracle


def _sul(piles=(1,)):
    env = NimEnv(piles=piles)
    oracle = MinimaxPreferenceOracle(env, largest_pile_heuristic, depth=None)
    return env, PreferenceSUL(env, oracle)


def test_finds_disagreement_against_wrong_hypothesis():
    env, sul = _sul(piles=(1, 2, 3))
    # A one-state machine that emits a fixed (wrong) output on every input.
    only_action = env.p1_alphabet[0]
    q = MealyState("q0")
    for a in env.p1_alphabet:
        q.transitions[a] = (only_action, q)
    bogus = MealyMachine([q], q)

    eq = BoundedExactOracle(env.p1_alphabet, sul, max_length=2)
    cex = eq.find_counterexample(bogus)
    assert cex is not None
    assert bogus.output_of(cex) != list(sul.query(cex))


def test_no_counterexample_when_single_input_sink():
    # Nim (1,): the only input ends the game -> SUL emits a sink (None).
    env, sul = _sul(piles=(1,))
    q = MealyState("q0")
    q.transitions[(0, 1)] = (None, q)
    correct = MealyMachine([q], q)
    eq = BoundedExactOracle(env.p1_alphabet, sul, max_length=3)
    assert eq.find_counterexample(correct) is None
