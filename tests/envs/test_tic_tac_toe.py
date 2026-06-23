"""Unit tests for the Tic-Tac-Toe environment."""

from pals.envs.base import Environment, Player
from pals.envs.tic_tac_toe import (
    EMPTY,
    P1_MARK,
    P2_MARK,
    TicTacToeEnv,
    TTTState,
    line_control_heuristic,
)


def test_is_an_environment():
    assert isinstance(TicTacToeEnv(), Environment)


def test_initial_state_is_empty_p1_to_move():
    env = TicTacToeEnv()
    s = env.initial_state()
    assert s.board == (EMPTY,) * 9
    assert s.player is Player.P1
    assert env.legal_actions(s) == list(range(9))


def test_step_places_mark_and_flips_player():
    env = TicTacToeEnv()
    s = env.step(env.initial_state(), 4)
    assert s.board[4] == P1_MARK
    assert s.player is Player.P2
    assert 4 not in env.legal_actions(s)


def test_p1_wins_gives_negative_reward():
    # X occupies the top row.
    board = (P1_MARK, P1_MARK, P1_MARK, P2_MARK, P2_MARK, EMPTY, EMPTY, EMPTY, EMPTY)
    state = TTTState(board=board, player=Player.P2)
    env = TicTacToeEnv()
    assert env.is_terminal(state)
    assert env.reward(state) == -1.0


def test_p2_wins_gives_positive_reward():
    board = (P2_MARK, P2_MARK, P2_MARK, P1_MARK, P1_MARK, EMPTY, EMPTY, EMPTY, EMPTY)
    state = TTTState(board=board, player=Player.P1)
    env = TicTacToeEnv()
    assert env.reward(state) == 1.0


def test_full_board_draw_is_terminal_zero():
    board = (
        P1_MARK,
        P2_MARK,
        P1_MARK,
        P1_MARK,
        P2_MARK,
        P2_MARK,
        P2_MARK,
        P1_MARK,
        P1_MARK,
    )
    state = TTTState(board=board, player=Player.P1)
    env = TicTacToeEnv()
    assert env.is_terminal(state)
    assert env.reward(state) == 0.0


def test_heuristic_sign_favours_p2_lines():
    env = TicTacToeEnv()
    p2_line = TTTState(
        board=(P2_MARK, P2_MARK, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY),
        player=Player.P1,
    )
    p1_line = TTTState(
        board=(P1_MARK, P1_MARK, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY),
        player=Player.P2,
    )
    assert line_control_heuristic(env, p2_line) > 0
    assert line_control_heuristic(env, p1_line) < 0


def test_exact_oracle_takes_the_winning_move():
    from pals.core.preference import MinimaxPreferenceOracle

    env = TicTacToeEnv()
    # O (P2) to move with two-in-a-row at 0,1; the winning move is cell 2.
    state = TTTState(
        board=(P2_MARK, P2_MARK, EMPTY, P1_MARK, P1_MARK, EMPTY, EMPTY, EMPTY, EMPTY),
        player=Player.P2,
    )
    oracle = MinimaxPreferenceOracle(env, line_control_heuristic, depth=None)
    assert oracle._value(state, None) == 1.0  # O wins under optimal play
    best = max(
        env.legal_actions(state),
        key=lambda a: oracle._value(env.step(state, a), None),
    )
    assert best == 2
