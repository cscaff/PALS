"""Dots & Boxes as a PALS :class:`Environment`.

An ``rows x cols`` grid of boxes; players alternately draw edges, scoring a point
for each box they complete. Most boxes wins; reward is from P2's perspective
(``+1`` win / ``-1`` loss / ``0`` tie).

**Simplification — strict alternation.** Standard Dots & Boxes grants an extra
turn when you complete a box, which breaks the strict P1/P2 alternation the SUL
and L* assume. We deliberately drop that rule: the turn always passes. This keeps
the game well-defined inside the two-player framing (and avoids the alternation
complexity that made the old implementation unwieldy) while preserving the
combinatorial, chain-building flavour. Edges are numbered, and the full edge set
is the L* input alphabet (legal moves only shrink).
"""

from __future__ import annotations

from typing import NamedTuple

from pals.envs.base import Action, Environment, Player


class DBState(NamedTuple):
    drawn: frozenset[int]
    p1_score: int
    p2_score: int
    player: Player


class DotsAndBoxesEnv(Environment):
    def __init__(self, rows: int = 1, cols: int = 2) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        self.rows = rows
        self.cols = cols

        # Edge numbering: horizontal edges first, then vertical.
        self._n_horizontal = (rows + 1) * cols
        n_vertical = rows * (cols + 1)
        self.n_edges = self._n_horizontal + n_vertical

        # Precompute the 4 edges bounding each box and the boxes each edge borders.
        self._box_edges: list[tuple[int, int, int, int]] = []
        self._boxes_of_edge: dict[int, list[int]] = {e: [] for e in range(self.n_edges)}
        for br in range(rows):
            for bc in range(cols):
                top = br * cols + bc
                bottom = (br + 1) * cols + bc
                left = self._n_horizontal + br * (cols + 1) + bc
                right = self._n_horizontal + br * (cols + 1) + (bc + 1)
                box = len(self._box_edges)
                self._box_edges.append((top, bottom, left, right))
                for e in (top, bottom, left, right):
                    self._boxes_of_edge[e].append(box)

    @property
    def n_boxes(self) -> int:
        return self.rows * self.cols

    @property
    def p1_alphabet(self) -> list[Action]:
        return list(range(self.n_edges))

    def initial_state(self) -> DBState:
        return DBState(frozenset(), 0, 0, Player.P1)

    def current_player(self, state: DBState) -> Player:
        return state.player

    def legal_actions(self, state: DBState) -> list[Action]:
        if self.is_terminal(state):
            return []
        return [e for e in range(self.n_edges) if e not in state.drawn]

    def step(self, state: DBState, action: Action) -> DBState:
        drawn = state.drawn | {action}
        completed = 0
        for box in self._boxes_of_edge[action]:
            edges = self._box_edges[box]
            if all(e in drawn for e in edges) and not all(
                e in state.drawn for e in edges
            ):
                completed += 1

        p1, p2 = state.p1_score, state.p2_score
        if state.player is Player.P1:
            p1 += completed
        else:
            p2 += completed
        nxt = Player.P2 if state.player is Player.P1 else Player.P1
        return DBState(frozenset(drawn), p1, p2, nxt)

    def is_terminal(self, state: DBState) -> bool:
        return len(state.drawn) == self.n_edges

    def reward(self, state: DBState) -> float:
        diff = state.p2_score - state.p1_score
        if diff > 0:
            return 1.0
        if diff < 0:
            return -1.0
        return 0.0


def score_margin_heuristic(env: DotsAndBoxesEnv, state: DBState) -> float:
    """A suboptimal greedy heuristic: the current score margin from P2's
    perspective, normalized by the number of boxes. Chases immediate boxes and
    ignores chain/sacrifice strategy."""
    return (state.p2_score - state.p1_score) / env.n_boxes
