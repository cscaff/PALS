"""Preference oracles — the "teacher" PALS learns against.

A preference oracle encodes a user's notion of *good play* for the system (P2).
It is deliberately allowed to be **locally** rather than globally optimal: PALS's
job is to recover globally preferred behaviour the oracle misses, via the MCTS
audit. An oracle exposes two operations:

* ``preferred_move(trace)`` — the P2 action the oracle greedily favours at a
  P2-turn prefix. Drives membership queries.
* ``compare(trace1, trace2)`` — a pairwise preference over two traces. Drives the
  MCTS audit's pairwise voting.

:class:`MinimaxPreferenceOracle` is a reusable implementation: depth-limited
minimax over an :class:`~pals.envs.base.Environment`, parametrized by a heuristic
used at the depth cutoff. A shallow depth + a greedy heuristic yields exactly the
locally-optimal-but-globally-suboptimal teacher PALS is designed to improve upon.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, auto
from typing import Protocol, runtime_checkable

from pals.envs.base import Action, Environment, Player, State


class Preference(Enum):
    """Outcome of comparing two traces."""

    FIRST = auto()  # trace1 is preferred
    SECOND = auto()  # trace2 is preferred
    EQUAL = auto()  # no preference


@runtime_checkable
class PreferenceOracle(Protocol):
    """Structural interface every preference oracle satisfies."""

    def preferred_move(self, trace: Sequence[Action]) -> Action | None:
        """The P2 action favoured at ``trace`` (``None`` if not a P2 turn)."""
        ...

    def compare(self, trace1: Sequence[Action], trace2: Sequence[Action]) -> Preference:
        """Pairwise preference between two traces."""
        ...


# A heuristic estimates a state's value (P2's perspective) at the depth cutoff.
class Heuristic(Protocol):
    def __call__(self, env: Environment, state: State) -> float: ...


class MinimaxPreferenceOracle:
    """Depth-limited minimax preference oracle over an ``Environment``.

    P2 maximizes and P1 minimizes the value (P2's perspective). With
    ``depth=None`` this is exact minimax (a *globally* optimal oracle); with a
    finite ``depth`` and a greedy ``heuristic`` it is locally optimal — the
    realistic, suboptimal teacher PALS aims to improve on.
    """

    def __init__(
        self,
        env: Environment,
        heuristic: Heuristic,
        depth: int | None = None,
    ) -> None:
        self.env = env
        self.heuristic = heuristic
        self.depth = depth
        self._cache: dict[tuple[State, int | None], float] = {}

    def _value(self, state: State, depth: int | None) -> float:
        key = (state, depth)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        env = self.env
        if env.is_terminal(state):
            result = env.reward(state)
        elif depth == 0:
            result = self.heuristic(env, state)
        else:
            next_depth = None if depth is None else depth - 1
            child_values = [
                self._value(env.step(state, a), next_depth)
                for a in env.legal_actions(state)
            ]
            if env.current_player(state) is Player.P2:
                result = max(child_values)
            else:
                result = min(child_values)

        self._cache[key] = result
        return result

    def _trace_value(self, trace: Sequence[Action]) -> float:
        state = self.env.get_node(trace)
        if state is None:
            return float("-inf")  # off-tree traces are maximally dispreferred
        return self._value(state, self.depth)

    def preferred_move(self, trace: Sequence[Action]) -> Action | None:
        env = self.env
        state = env.get_node(trace)
        if (
            state is None
            or env.is_terminal(state)
            or env.current_player(state) is not Player.P2
        ):
            return None
        next_depth = None if self.depth is None else self.depth - 1
        return max(
            env.legal_actions(state),
            key=lambda a: self._value(env.step(state, a), next_depth),
        )

    def compare(self, trace1: Sequence[Action], trace2: Sequence[Action]) -> Preference:
        v1 = self._trace_value(trace1)
        v2 = self._trace_value(trace2)
        if v1 > v2:
            return Preference.FIRST
        if v2 > v1:
            return Preference.SECOND
        return Preference.EQUAL
