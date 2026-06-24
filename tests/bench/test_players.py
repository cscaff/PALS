"""Unit tests for benchmark players."""

import random

from pals.bench.players import (
    PALSPlayer,
    QLearningPlayer,
    RandomPlayer,
    UCTPlayer,
    greedy_player,
    optimal_player,
)
from pals.core.learner import run_pals
from pals.core.preference import MinimaxPreferenceOracle
from pals.envs.nim import NimEnv, largest_pile_heuristic


def _nim():
    return NimEnv(piles=(1, 2, 3))


def test_random_player_picks_legal_action():
    env = _nim()
    rng = random.Random(0)
    a = RandomPlayer().action(env, [], rng)
    assert a in env.legal_actions(env.initial_state())


def test_optimal_p2_beats_optimal_p1_on_losing_first_move():
    # Nim (1,2,3): nim-sum 0, so the player to move (P1) loses -> P2 wins.
    env = _nim()
    from pals.bench.evaluate import play_game

    opt = optimal_player(largest_pile_heuristic)
    assert play_game(env, opt, opt, random.Random(0)) == 1.0


def test_greedy_and_optimal_return_legal_actions():
    env = _nim()
    rng = random.Random(1)
    for player in (
        greedy_player(largest_pile_heuristic),
        optimal_player(largest_pile_heuristic),
    ):
        assert player.action(env, [], rng) in env.legal_actions(env.initial_state())


def test_uct_player_returns_legal_action():
    env = _nim()
    a = UCTPlayer(budget=30).action(env, [], random.Random(2))
    assert a in env.legal_actions(env.initial_state())


def test_qlearning_player_returns_legal_action():
    env = _nim()
    player = QLearningPlayer(env, episodes=200, rng=random.Random(3))
    assert player.action(env, [], random.Random(0)) in env.legal_actions(
        env.initial_state()
    )


def test_pals_player_returns_legal_action():
    env = _nim()
    oracle = MinimaxPreferenceOracle(env, largest_pile_heuristic, depth=None)
    result = run_pals(env, oracle, depth_n=3, rollout_budget=20, rng=random.Random(0))
    player = PALSPlayer(result.model)
    assert player.action(env, [], random.Random(0)) in env.legal_actions(
        env.initial_state()
    )
