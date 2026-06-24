"""Unit tests for the noisy preference oracle."""

import pytest

from pals.bench.noisy import NoisyOracle
from pals.core.preference import MinimaxPreferenceOracle, Preference
from pals.envs.nim import NimEnv, largest_pile_heuristic


def _setup():
    env = NimEnv(piles=(1, 2, 3))
    inner = MinimaxPreferenceOracle(env, largest_pile_heuristic, depth=None)
    return env, inner


def test_rejects_bad_noise():
    env, inner = _setup()
    with pytest.raises(ValueError):
        NoisyOracle(inner, env, noise=1.5)


def test_zero_noise_matches_inner():
    env, inner = _setup()
    noisy = NoisyOracle(inner, env, noise=0.0, seed=0)
    assert noisy.preferred_move([(0, 1)]) == inner.preferred_move([(0, 1)])
    assert noisy.compare([(0, 1)], [(1, 1)]) == inner.compare([(0, 1)], [(1, 1)])


def test_full_noise_still_returns_valid_outputs():
    env, inner = _setup()
    noisy = NoisyOracle(inner, env, noise=1.0, seed=1)
    move = noisy.preferred_move([(0, 1)])
    assert move in env.p2_legal_moves([(0, 1)])
    assert noisy.compare([(0, 1)], [(1, 1)]) in set(Preference)


def test_corruption_is_deterministic_in_the_input():
    env, inner = _setup()
    noisy = NoisyOracle(inner, env, noise=0.5, seed=7)
    # Same query always yields the same answer (a fixed imperfect oracle), so
    # the SUL stays deterministic and L* can converge.
    answers = [noisy.preferred_move([(0, 1)]) for _ in range(10)]
    assert len(set(answers)) == 1
    # Two instances with the same seed agree everywhere.
    twin = NoisyOracle(inner, env, noise=0.5, seed=7)
    assert noisy.preferred_move([(1, 1)]) == twin.preferred_move([(1, 1)])
