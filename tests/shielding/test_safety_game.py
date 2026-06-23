"""Unit tests for the generic safety-game solver."""

from _graph_env import GraphEnv

from pals.envs.base import Player
from pals.envs.gas_grid import (
    REFUEL,
    GasGridEnv,
    GasGridState,
    gas_depleted,
    safety_state_key,
)
from pals.shielding.safety_game import solve_safety_game


def test_p1_state_losing_if_any_successor_bad():
    # 0 (P1) -> a:safe-sink(1), b:bad(2). Adversarial P1 can pick b, so 0 loses.
    env = GraphEnv(
        start=0,
        players={0: Player.P1, 1: Player.P2, 2: Player.P2},
        edges={0: {"a": 1, "b": 2}},
        terminals={1, 2},
    )
    game = solve_safety_game(env, is_bad=lambda s: s == 2)
    assert 0 not in game.winning
    assert 2 not in game.winning
    assert 1 in game.winning  # safe terminal


def test_p2_state_winning_if_some_successor_safe():
    # 0 (P2) -> a:safe-sink(1), b:bad(2). Cooperative P2 picks a, so 0 wins.
    env = GraphEnv(
        start=0,
        players={0: Player.P2, 1: Player.P2, 2: Player.P2},
        edges={0: {"a": 1, "b": 2}},
        terminals={1, 2},
    )
    game = solve_safety_game(env, is_bad=lambda s: s == 2)
    assert 0 in game.winning
    assert game.safe_action(0) == "a"


def test_prefer_action_chosen_when_safe():
    env = GraphEnv(
        start=0,
        players={0: Player.P2, 1: Player.P2, 2: Player.P2},
        edges={0: {"a": 1, "b": 1}},  # both safe
        terminals={1, 2},
    )
    game = solve_safety_game(env, is_bad=lambda s: s == 2, prefer_action=lambda s: "b")
    assert game.safe_action(0) == "b"


def test_gas_grid_strategy_refuels_before_running_dry():
    # 1x3 corridor, gas non-fatal so the spec is the only thing forbidding gas<=0.
    env = GasGridEnv(
        rows=1,
        cols=3,
        home=(0, 0),
        refuel=(0, 1),
        dropoff=(0, 2),
        gas_max=2,
        eligible_cells=((0, 0),),
        gas_is_fatal=False,
    )
    game = solve_safety_game(env, is_bad=gas_depleted, state_key=safety_state_key)
    assert safety_state_key(env.initial_state()) in game.winning
    # At (0,1) carrying with one gas left, the only safe move is to refuel.
    low_gas = GasGridState(
        pos=(0, 1),
        gas=1,
        task_loc=(0, 0),
        carrying=True,
        delivered=False,
        player=Player.P2,
        step_count=3,
    )
    assert game.safe_action(safety_state_key(low_gas)) == REFUEL
