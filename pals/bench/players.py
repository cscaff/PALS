"""Players for benchmarking — shared across all environments.

A *player* chooses an action given the interleaved trace so far. The same player
classes serve as P1 (environment) opponents or P2 (system) controllers; each uses
``current_player`` to know whether to maximize (P2) or minimize (P1) the
P2-perspective value, so a single implementation works on either side.

Players: :class:`RandomPlayer`, :class:`MinimaxPlayer` (``Greedy`` = depth 1 over a
heuristic, ``Optimal`` = exact), :class:`UCTPlayer`, :class:`QLearningPlayer`, and
:class:`PALSPlayer` (wraps a learned Mealy machine). They are the baselines PALS
is compared against, plus the adapter that lets a learned controller play.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Protocol

from pals.core.lstar import MealyMachine
from pals.core.preference import Heuristic
from pals.envs.base import Action, Environment, Player, State


class GamePlayer(Protocol):
    def action(self, env: Environment, trace: Sequence[Action], rng: random.Random):
        """Choose an action at the state reached by ``trace``."""
        ...


def _p1_inputs(trace: Sequence[Action]) -> list[Action]:
    return [trace[i] for i in range(0, len(trace), 2)]


class RandomPlayer:
    def action(self, env, trace, rng):
        return rng.choice(env.legal_actions(env.get_node(trace)))


# ----------------------------------------------------------------------
# Minimax players (Greedy = shallow + heuristic, Optimal = exact)
# ----------------------------------------------------------------------


def _minimax_value(
    env: Environment,
    state: State,
    depth: int | None,
    heuristic: Heuristic,
    cache: dict,
) -> float:
    key = (state, depth)
    if key in cache:
        return cache[key]
    if env.is_terminal(state):
        v = env.reward(state)
    elif depth == 0:
        v = heuristic(env, state)
    else:
        nd = None if depth is None else depth - 1
        children = [
            _minimax_value(env, env.step(state, a), nd, heuristic, cache)
            for a in env.legal_actions(state)
        ]
        v = max(children) if env.current_player(state) is Player.P2 else min(children)
    cache[key] = v
    return v


class MinimaxPlayer:
    """Picks the action optimizing the (depth-limited) minimax value for the
    side to move. ``depth=None`` is exact (Optimal); ``depth=1`` is Greedy."""

    def __init__(self, heuristic: Heuristic, depth: int | None = None) -> None:
        self.heuristic = heuristic
        self.depth = depth
        self._cache: dict = {}

    def action(self, env, trace, rng):
        state = env.get_node(trace)
        nd = None if self.depth is None else self.depth - 1
        scored = [
            (
                a,
                _minimax_value(
                    env, env.step(state, a), nd, self.heuristic, self._cache
                ),
            )
            for a in env.legal_actions(state)
        ]
        pick = max if env.current_player(state) is Player.P2 else min
        target = pick(v for _, v in scored)
        return next(a for a, v in scored if v == target)


def greedy_player(heuristic: Heuristic) -> MinimaxPlayer:
    return MinimaxPlayer(heuristic, depth=1)


def optimal_player(heuristic: Heuristic) -> MinimaxPlayer:
    return MinimaxPlayer(heuristic, depth=None)


def greedy_action(env: Environment, heuristic: Heuristic):
    """A *state -> action* one-ply greedy policy. Useful as the shield's
    ``prefer_action`` tie-breaker so the safe strategy keeps the controller's
    preferred move wherever it is safe, falling back only where it is not."""

    def choose(state: State) -> Action | None:
        if env.is_terminal(state):
            return None
        actions = env.legal_actions(state)
        if not actions:
            return None
        pick = max if env.current_player(state) is Player.P2 else min
        return pick(actions, key=lambda a: heuristic(env, env.step(state, a)))

    return choose


# ----------------------------------------------------------------------
# UCT (minimax-flavoured Monte Carlo tree search)
# ----------------------------------------------------------------------


class UCTPlayer:
    def __init__(self, budget: int = 200, c: float = 1.4) -> None:
        self.budget = budget
        self.c = c

    def action(self, env, trace, rng):
        root = env.get_node(trace)
        visits: dict[State, int] = {}
        value: dict[State, float] = {}  # mean P2-perspective reward

        for _ in range(self.budget):
            self._simulate(env, root, visits, value, rng)

        # Choose the most-visited child, broken by value for the side to move.
        best_for = max if env.current_player(root) is Player.P2 else min
        actions = env.legal_actions(root)
        return best_for(
            actions,
            key=lambda a: (
                visits.get(env.step(root, a), 0),
                value.get(env.step(root, a), 0.0)
                if env.current_player(root) is Player.P2
                else -value.get(env.step(root, a), 0.0),
            ),
        )

    def _simulate(self, env, state, visits, value, rng) -> float:
        if env.is_terminal(state):
            return env.reward(state)

        actions = env.legal_actions(state)
        children = [env.step(state, a) for a in actions]
        unvisited = [c for c in children if c not in visits]
        if unvisited:
            child = rng.choice(unvisited)
            reward = self._rollout(env, child, rng)
        else:
            total = sum(visits[c] for c in children)
            is_p2 = env.current_player(state) is Player.P2
            child = (max if is_p2 else min)(
                children, key=lambda c: self._ucb(c, total, visits, value, is_p2)
            )
            reward = self._simulate(env, child, visits, value, rng)

        visits[child] = visits.get(child, 0) + 1
        value[child] = (
            value.get(child, 0.0) + (reward - value.get(child, 0.0)) / visits[child]
        )
        return reward

    def _ucb(self, child, total, visits, value, is_p2) -> float:
        n = visits[child]
        exploit = value[child] if is_p2 else -value[child]
        return exploit + self.c * math.sqrt(math.log(total + 1) / n)

    def _rollout(self, env, state, rng) -> float:
        while not env.is_terminal(state):
            state = env.step(state, rng.choice(env.legal_actions(state)))
        return env.reward(state)


# ----------------------------------------------------------------------
# Tabular Q-learning (P2 learns; P1 plays randomly during training)
# ----------------------------------------------------------------------


class QLearningPlayer:
    def __init__(
        self,
        env: Environment,
        episodes: int = 3000,
        alpha: float = 0.5,
        gamma: float = 0.95,
        eps: float = 0.2,
        rng: random.Random | None = None,
    ) -> None:
        self.q: dict[tuple[State, Action], float] = {}
        rng = rng or random.Random()
        for _ in range(episodes):
            self._episode(env, alpha, gamma, eps, rng)

    def _q(self, state, action) -> float:
        return self.q.get((state, action), 0.0)

    def _best(self, env, state) -> tuple[Action, float]:
        actions = env.legal_actions(state)
        is_p2 = env.current_player(state) is Player.P2
        pick = max if is_p2 else min
        best_a = pick(actions, key=lambda a: self._q(state, a))
        return best_a, self._q(state, best_a)

    def _episode(self, env, alpha, gamma, eps, rng) -> None:
        state = env.initial_state()
        while not env.is_terminal(state):
            actions = env.legal_actions(state)
            is_p2 = env.current_player(state) is Player.P2
            if not is_p2:
                # Environment plays randomly during training.
                state = env.step(state, rng.choice(actions))
                continue
            action = (
                rng.choice(actions)
                if rng.random() < eps
                else max(actions, key=lambda a: self._q(state, a))
            )
            nxt = env.step(state, action)
            # Bootstrap from the next P2 decision (or the terminal reward).
            future = self._lookahead(env, nxt)
            target = future
            old = self._q(state, action)
            self.q[(state, action)] = old + alpha * (gamma * target - old)
            state = nxt

    def _lookahead(self, env, state) -> float:
        """Value backed up to the previous P2 move: skip P1 (random) moves and
        return the next P2 state's best Q, or the terminal reward."""
        while not env.is_terminal(state) and env.current_player(state) is Player.P1:
            # Expected over the (random) environment is approximated by the best
            # case here; training episodes supply the averaging.
            return self._lookahead(env, env.step(state, env.legal_actions(state)[0]))
        if env.is_terminal(state):
            return env.reward(state)
        return self._best(env, state)[1]

    def action(self, env, trace, rng):
        state = env.get_node(trace)
        actions = env.legal_actions(state)
        is_p2 = env.current_player(state) is Player.P2
        pick = max if is_p2 else min
        return pick(actions, key=lambda a: self._q(state, a))


# ----------------------------------------------------------------------
# PALS controller (wraps a learned Mealy machine)
# ----------------------------------------------------------------------


class PALSPlayer:
    def __init__(self, model: MealyMachine) -> None:
        self.model = model

    def action(self, env, trace, rng):
        state = env.get_node(trace)
        legal = env.legal_actions(state)
        outputs = self.model.output_of(_p1_inputs(trace))
        action = outputs[-1] if outputs else None
        if action in legal:
            return action
        return rng.choice(legal)  # off-model fallback keeps play legal
