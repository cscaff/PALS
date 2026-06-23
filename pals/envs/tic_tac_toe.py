"""Tic-Tac-Toe as a PALS :class:`Environment`.

P1 plays ``X`` and moves first; P2 (the system PALS learns) plays ``O``. An
action is a board cell index ``0..8``; rewards are from P2's perspective
(``+1`` if O wins, ``-1`` if X wins, ``0`` for a draw). The full cell set
``0..8`` is the L* input alphabet (legal moves only shrink).
"""

from __future__ import annotations

from typing import NamedTuple

from pals.envs.base import Action, Environment, Player

EMPTY = ""
P1_MARK = "X"
P2_MARK = "O"

_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),  # rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # cols
    (0, 4, 8),
    (2, 4, 6),  # diagonals
)


class TTTState(NamedTuple):
    board: tuple[str, ...]
    player: Player


def _mark(player: Player) -> str:
    return P1_MARK if player is Player.P1 else P2_MARK


def _winner(board: tuple[str, ...]) -> str | None:
    for a, b, c in _LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return None


class TicTacToeEnv(Environment):
    @property
    def p1_alphabet(self) -> list[Action]:
        return list(range(9))

    def initial_state(self) -> TTTState:
        return TTTState(board=(EMPTY,) * 9, player=Player.P1)

    def current_player(self, state: TTTState) -> Player:
        return state.player

    def legal_actions(self, state: TTTState) -> list[Action]:
        if self.is_terminal(state):
            return []
        return [i for i, cell in enumerate(state.board) if cell == EMPTY]

    def step(self, state: TTTState, action: Action) -> TTTState:
        board = list(state.board)
        board[action] = _mark(state.player)
        nxt = Player.P2 if state.player is Player.P1 else Player.P1
        return TTTState(board=tuple(board), player=nxt)

    def is_terminal(self, state: TTTState) -> bool:
        return _winner(state.board) is not None or EMPTY not in state.board

    def reward(self, state: TTTState) -> float:
        winner = _winner(state.board)
        if winner == P2_MARK:
            return 1.0
        if winner == P1_MARK:
            return -1.0
        return 0.0


def line_control_heuristic(env: TicTacToeEnv, state: TTTState) -> float:
    """A suboptimal static eval: each line counts toward whichever mark solely
    occupies it, summed from P2's perspective and scaled into ``(-1, 1)``."""
    board = state.board
    score = 0
    for a, b, c in _LINES:
        cells = (board[a], board[b], board[c])
        has_p1 = P1_MARK in cells
        has_p2 = P2_MARK in cells
        if has_p2 and not has_p1:
            score += cells.count(P2_MARK)
        elif has_p1 and not has_p2:
            score -= cells.count(P1_MARK)
    return score / 9.0
