"""Unit tests for game playing and evaluation."""

import random

from pals.bench.evaluate import evaluate, evaluate_vs_opponents, play_game
from pals.bench.players import RandomPlayer, greedy_player, optimal_player
from pals.envs.nim import NimEnv, largest_pile_heuristic


def _nim():
    return NimEnv(piles=(1, 2, 3))


def test_play_game_returns_reward_in_range():
    env = _nim()
    r = play_game(env, RandomPlayer(), RandomPlayer(), random.Random(0))
    assert r in (-1.0, 1.0)


def test_evaluate_deterministic_players_are_consistent():
    env = _nim()
    opt = optimal_player(largest_pile_heuristic)
    a = evaluate(env, opt, opt, n_games=3, rng=random.Random(0))
    b = evaluate(env, opt, opt, n_games=3, rng=random.Random(0))
    assert a == b == 1.0  # P2 wins (nim-sum 0 -> first mover loses)


def test_evaluate_vs_opponents_has_all_columns():
    env = _nim()
    opponents = {
        "vs_random": RandomPlayer(),
        "vs_greedy": greedy_player(largest_pile_heuristic),
        "vs_optimal": optimal_player(largest_pile_heuristic),
    }
    scores = evaluate_vs_opponents(
        env,
        optimal_player(largest_pile_heuristic),
        opponents,
        n_games=5,
        rng=random.Random(0),
    )
    assert set(scores) == set(opponents)
    assert all(-1.0 <= v <= 1.0 for v in scores.values())
