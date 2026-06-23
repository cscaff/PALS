"""Nim as a PALS :class:`Environment`.

Normal-play Nim: players alternately remove one or more objects from a single
pile; the player who removes the last object **wins**. P1 moves first.

State is an immutable ``(piles, player)`` pair. An action is a
``(pile_index, count)`` pair meaning "remove ``count`` objects from pile
``pile_index``". Because piles only ever shrink and the action encoding is
position-based, the set of P1 actions legal at the root is a superset of the
P1 actions legal anywhere — so it serves as the L* input alphabet.
"""

from __future__ import annotations

from typing import NamedTuple

from pals.envs.base import Action, Environment, Player

PileIndex = int
Count = int
NimAction = tuple[PileIndex, Count]


class NimState(NamedTuple):
    piles: tuple[int, ...]
    player: Player


def _apply(piles: tuple[int, ...], pile_index: int, count: int) -> tuple[int, ...]:
    return piles[:pile_index] + (piles[pile_index] - count,) + piles[pile_index + 1 :]


def _other(player: Player) -> Player:
    return Player.P2 if player is Player.P1 else Player.P1


class NimEnv(Environment):
    def __init__(self, piles: tuple[int, ...] = (1, 2, 3)) -> None:
        if any(p < 0 for p in piles):
            raise ValueError(f"pile sizes must be non-negative: {piles!r}")
        self._piles = tuple(piles)

    @property
    def p1_alphabet(self) -> list[Action]:
        # P1 moves first, so legal_actions at the root enumerates every P1 move
        # that can ever occur (piles only shrink, encoding is position-based).
        return self.legal_actions(self.initial_state())

    def initial_state(self) -> NimState:
        return NimState(piles=self._piles, player=Player.P1)

    def current_player(self, state: NimState) -> Player:
        return state.player

    def legal_actions(self, state: NimState) -> list[NimAction]:
        if self.is_terminal(state):
            return []
        return [
            (i, c) for i, size in enumerate(state.piles) for c in range(1, size + 1)
        ]

    def step(self, state: NimState, action: NimAction) -> NimState:
        pile_index, count = action
        return NimState(
            piles=_apply(state.piles, pile_index, count),
            player=_other(state.player),
        )

    def is_terminal(self, state: NimState) -> bool:
        return all(p == 0 for p in state.piles)

    def reward(self, state: NimState) -> float:
        # Normal play: the player to move at a terminal state has no objects to
        # take, so the *other* player took the last object and won. Reward is
        # from P2's perspective: +1 if P2 won, -1 if P1 won.
        winner = _other(state.player)
        return 1.0 if winner is Player.P2 else -1.0


def largest_pile_heuristic(env: NimEnv, state: NimState) -> float:
    """A deliberately *suboptimal* greedy heuristic for the preference oracle.

    Scores a state by how much the largest pile dominates the total, ignoring
    the nim-sum that actually determines the winner. This makes the oracle play
    plausibly but not optimally — exactly the locally-optimal teacher PALS is
    meant to improve upon. Scaled by 0.9 to stay strictly inside the terminal
    payoff range, and signed to P2's perspective.
    """
    total = sum(state.piles)
    if total == 0:
        return 0.0
    score = (max(state.piles) / total) * 0.9
    return score if state.player is Player.P2 else -score
