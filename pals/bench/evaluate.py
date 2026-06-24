"""Play games and score strategies.

``play_game`` drives one P1-vs-P2 game to termination and returns the terminal
reward (P2's perspective). ``evaluate`` averages over games (games only differ
when a player is stochastic). ``evaluate_vs_opponents`` scores a P2 controller
against several P1 opponents — the ``vs_random`` / ``vs_greedy`` / ``vs_optimal``
columns the paper reports.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

from pals.bench.players import GamePlayer
from pals.envs.base import Environment, Player


def play_game(
    env: Environment,
    p1_player: GamePlayer,
    p2_player: GamePlayer,
    rng: random.Random,
) -> float:
    """Play one game; return the terminal reward from P2's perspective."""
    trace: list = []
    state = env.initial_state()
    while not env.is_terminal(state):
        actor = p1_player if env.current_player(state) is Player.P1 else p2_player
        action = actor.action(env, trace, rng)
        trace.append(action)
        state = env.step(state, action)
    return env.reward(state)


def evaluate(
    env: Environment,
    p2_player: GamePlayer,
    p1_player: GamePlayer,
    n_games: int = 20,
    rng: random.Random | None = None,
) -> float:
    """Mean P2 reward of ``p2_player`` against ``p1_player`` over ``n_games``."""
    rng = rng or random.Random()
    return sum(play_game(env, p1_player, p2_player, rng) for _ in range(n_games)) / (
        n_games
    )


def evaluate_vs_opponents(
    env: Environment,
    p2_player: GamePlayer,
    opponents: Mapping[str, GamePlayer],
    n_games: int = 20,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Score ``p2_player`` against each named P1 opponent."""
    rng = rng or random.Random()
    return {
        name: evaluate(env, p2_player, opp, n_games=n_games, rng=rng)
        for name, opp in opponents.items()
    }
