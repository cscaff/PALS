"""Table B — the persistent MCTS exploration record.

While the L* observation table (Table A) records only the moves the preference
oracle *selected*, Table B records every alternative the audit has *considered*,
with UCB statistics. It persists across all equivalence queries (it accumulates;
it is never reset), so exploration knowledge carries over between rounds.

Keys are full action-traces (interleaved P1/P2). Each ``(trace, action)`` pair
tracks a visit count, a current value in ``[0, 1]`` (assigned by the SMT valuer),
and a "pruned" flag that permanently removes hopeless branches from sampling.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from pals.envs.base import Action

TraceKey = tuple[Action, ...]

# UCB score handed to an action that has never been visited, so unexplored
# alternatives are always tried before any explored one is exploited.
UNVISITED_SCORE = 0.75


@dataclass
class ActionStats:
    visits: int = 0
    value: float = 0.5  # neutral prior until the SMT valuer assigns a real value
    pruned: bool = False


class TableB:
    def __init__(self, exploration_c: float = 1.4, depth_alpha: float = 1.2) -> None:
        """``exploration_c`` scales the UCB bonus; ``depth_alpha`` (> 1) damps it
        at greater depth so shallow nodes are explored more aggressively."""
        self.c = exploration_c
        self.alpha = depth_alpha
        self._nodes: dict[TraceKey, dict[Action, ActionStats]] = {}

    def _stats(self, trace: Sequence[Action], action: Action) -> ActionStats:
        key = tuple(trace)
        bucket = self._nodes.setdefault(key, {})
        stats = bucket.get(action)
        if stats is None:
            stats = ActionStats()
            bucket[action] = stats
        return stats

    def actions_at(self, trace: Sequence[Action]) -> dict[Action, ActionStats]:
        """All actions recorded at ``trace`` (empty if none seen yet)."""
        return self._nodes.get(tuple(trace), {})

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def record_visit(self, trace: Sequence[Action], action: Action) -> None:
        stats = self._stats(trace, action)
        if not stats.pruned:
            stats.visits += 1

    def update_value(
        self, trace: Sequence[Action], action: Action, value: float
    ) -> None:
        self._stats(trace, action).value = value

    def prune(self, trace: Sequence[Action], action: Action) -> None:
        self._stats(trace, action).pruned = True

    # ------------------------------------------------------------------
    # UCB scoring & sampling
    # ------------------------------------------------------------------

    def ucb_score(
        self,
        trace: Sequence[Action],
        action: Action,
        available: Sequence[Action],
    ) -> float:
        """UCB score for ``(trace, action)`` among ``available`` siblings.

        Pruned actions score ``-inf``; unvisited ones score ``UNVISITED_SCORE``.
        """
        stats = self._stats(trace, action)
        if stats.pruned:
            return float("-inf")
        if stats.visits == 0:
            return UNVISITED_SCORE

        total = sum(
            self._stats(trace, a).visits
            for a in available
            if not self._stats(trace, a).pruned
        )
        if total == 0:
            return UNVISITED_SCORE

        depth_discount = self.alpha ** (-len(tuple(trace)))
        bonus = self.c * math.sqrt(math.log(total) / stats.visits)
        return stats.value + bonus * depth_discount

    def best_action(
        self, trace: Sequence[Action], available: Sequence[Action]
    ) -> Action | None:
        live = [a for a in available if not self._stats(trace, a).pruned]
        if not live:
            return None
        return max(live, key=lambda a: self.ucb_score(trace, a, available))

    def sample_action(
        self,
        trace: Sequence[Action],
        available: Sequence[Action],
        temperature: float = 1.0,
        rng: random.Random | None = None,
    ) -> Action | None:
        """Sample an action ~ softmax over UCB scores; unvisited get a high prior."""
        rng = rng or random
        live = [a for a in available if not self._stats(trace, a).pruned]
        if not live:
            return None

        weights = []
        for a in live:
            stats = self._stats(trace, a)
            if stats.visits == 0:
                weights.append(10.0)  # strong prior toward unexplored actions
            else:
                weights.append(math.exp(self.ucb_score(trace, a, live) / temperature))
        return rng.choices(live, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        states = len(self._nodes)
        edges = sum(len(b) for b in self._nodes.values())
        pruned = sum(s.pruned for b in self._nodes.values() for s in b.values())
        return f"TableB: {states} states, {edges} edges, {pruned} pruned"
