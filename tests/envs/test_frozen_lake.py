"""Unit tests for the FrozenLake environment."""

import pytest

from pals.envs.base import Player
from pals.envs.frozen_lake import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FrozenLakeEnv,
    in_hole_predicate,
    manhattan_progress_heuristic,
    safe_goal_action,
)


def test_rejects_non_rectangular_grid():
    with pytest.raises(ValueError):
        FrozenLakeEnv(desc=("SF", "FFG"))


def test_rejects_grid_without_single_goal():
    with pytest.raises(ValueError):
        FrozenLakeEnv(desc=("SF", "FF"))  # no goal


def test_parses_default_map():
    env = FrozenLakeEnv()
    assert env.rows == 4 and env.cols == 4
    assert env.goal == (3, 3)
    assert (1, 1) in env.holes and (3, 0) in env.holes


def test_start_cells_exclude_holes_goal_and_unreachable():
    env = FrozenLakeEnv()
    assert env.goal not in env.start_cells
    assert all(c not in env.holes for c in env.start_cells)


def test_initial_state_is_idle_p1():
    env = FrozenLakeEnv(desc=("SHG", "FFF"))
    s = env.initial_state()
    assert s.pos is None
    assert env.current_player(s) is Player.P1
    assert not env.is_terminal(s)


def test_p1_chooses_spawn_then_reveals_observation():
    env = FrozenLakeEnv(desc=("SHG", "FFF"))
    s = env.initial_state()
    spawns = env.legal_actions(s)
    assert all(a[0] == "SPAWN" for a in spawns)
    s2 = env.step(s, ("SPAWN", (0, 0)))
    assert s2.pos == (0, 0) and s2.player is Player.P2
    # After a P2 move it is P1's turn and the only input is the observation.
    s3 = env.step(s2, DOWN)
    assert s3.player is Player.P1
    assert env.legal_actions(s3) == [env.observation(s3)] == [(1, 0)]


def test_p2_offgrid_moves_are_illegal():
    env = FrozenLakeEnv(desc=("SHG", "FFF"))
    s = env.step(env.initial_state(), ("SPAWN", (0, 0)))  # corner, P2 to move
    actions = env.legal_actions(s)
    assert UP not in actions and LEFT not in actions  # off-grid
    assert RIGHT in actions and DOWN in actions


def test_goal_and_hole_terminality_and_reward():
    env = FrozenLakeEnv(desc=("SHG", "FFF"))  # holes fatal by default
    # Drive into the goal: spawn (0,2) is G — but goal excluded from spawns, so
    # build the goal/hole states directly via stepping.
    s = env.step(env.initial_state(), ("SPAWN", (0, 0)))
    s = env.step(s, RIGHT)  # -> (0,1) hole, P1 turn
    assert env.is_terminal(s)
    assert env.reward(s) == -1.0


def test_holes_non_fatal_makes_hole_passable():
    env = FrozenLakeEnv(desc=("SHG", "FFF"), holes_are_fatal=False)
    s = env.step(env.initial_state(), ("SPAWN", (0, 0)))
    s = env.step(s, RIGHT)  # step onto the hole (0,1)
    assert s.pos == (0, 1)
    assert not env.is_terminal(s)  # passable now
    assert in_hole_predicate(env)(s)  # but flagged unsafe


def test_heuristic_is_hole_blind():
    env = FrozenLakeEnv(desc=("SHG", "FFF"), holes_are_fatal=False)
    s = env.step(env.initial_state(), ("SPAWN", (0, 0)))
    on_hole = env.step(s, RIGHT)  # (0,1), one step from goal
    # Hole-blind: it scores the near-goal hole highly, not as a hazard.
    assert manhattan_progress_heuristic(env, on_hole) > 0.0


def test_safe_goal_action_routes_around_the_hole():
    env = FrozenLakeEnv(desc=("SHG", "FFF"), holes_are_fatal=False)
    choose = safe_goal_action(env)
    s = env.step(env.initial_state(), ("SPAWN", (0, 0)))
    assert choose(s) == DOWN  # detour, never RIGHT into the hole
