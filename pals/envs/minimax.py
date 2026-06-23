"""Random-valued minimax game trees as a PALS :class:`Environment`.

A complete ``branching``-ary tree of fixed ``depth``; leaves carry random
payoffs. Players alternate from the root (P1 first), P1 minimizing and P2
maximizing the payoff (P2's perspective). This is the benchmark where the audit's
depth-``N`` rollouts matter most (``N`` << tree depth), mirroring the paper's
minimax experiments.

A state is the path from the root (a tuple of child indices), so states are
hashable and transitions are computed without materializing node objects. Leaf
payoffs are drawn once at construction from a seeded RNG for reproducibility.
"""

from __future__ import annotations

import itertools
import random

from pals.envs.base import Action, Environment, Player

Path = tuple[int, ...]


class MinimaxEnv(Environment):
    def __init__(
        self,
        depth: int,
        branching: int = 2,
        seed: int = 0,
        value_range: tuple[float, float] = (0.0, 100.0),
    ) -> None:
        if depth < 1 or branching < 1:
            raise ValueError("depth and branching must be >= 1")
        self.depth = depth
        self.branching = branching
        rng = random.Random(seed)
        lo, hi = value_range
        self._leaf_values: dict[Path, float] = {
            path: rng.uniform(lo, hi)
            for path in itertools.product(range(branching), repeat=depth)
        }

    @property
    def p1_alphabet(self) -> list[Action]:
        return list(range(self.branching))

    def initial_state(self) -> Path:
        return ()

    def current_player(self, state: Path) -> Player:
        # P1 moves first at the root (even depth); players alternate.
        return Player.P1 if len(state) % 2 == 0 else Player.P2

    def legal_actions(self, state: Path) -> list[Action]:
        if self.is_terminal(state):
            return []
        return list(range(self.branching))

    def step(self, state: Path, action: Action) -> Path:
        return (*state, action)

    def is_terminal(self, state: Path) -> bool:
        return len(state) == self.depth

    def reward(self, state: Path) -> float:
        return self._leaf_values[state]


def leftmost_leaf_heuristic(env: MinimaxEnv, state: Path) -> float:
    """A suboptimal greedy heuristic: descend always taking the first action to
    a leaf and return its payoff, ignoring the opponent entirely."""
    s = state
    while not env.is_terminal(s):
        s = env.step(s, env.legal_actions(s)[0])
    return env.reward(s)
