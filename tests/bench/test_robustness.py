"""Robustness: PALS still learns a valid controller from a noisy oracle."""

import random

import pytest

from pals.bench.noisy import NoisyOracle
from pals.core.learner import run_pals
from pals.core.preference import MinimaxPreferenceOracle
from pals.envs.nim import NimEnv, largest_pile_heuristic


@pytest.mark.parametrize("noise", [0.0, 0.25, 0.5])
def test_pals_runs_under_noisy_oracle(noise):
    # The MCTS audit's termination assumes consistent (history-independent)
    # preferences, which noise violates; the L*+PAC path learns the imperfect
    # target faithfully and terminates, which is what we check for robustness.
    env = NimEnv(piles=(1, 2, 3))
    inner = MinimaxPreferenceOracle(env, largest_pile_heuristic, depth=1)
    oracle = NoisyOracle(inner, env, noise=noise, seed=0)
    result = run_pals(env, oracle, use_mcts=False, use_pac=True, rng=random.Random(0))
    # Even under a noisy teacher, the result is a valid, total Mealy machine.
    assert len(result.model.states) >= 1
    for state in result.model.states:
        assert set(state.transitions) == set(env.p1_alphabet)
