"""Unit and e2e tests for the gas-grid delivery environment."""

import random

from pals.envs.base import Environment, Player
from pals.envs.gas_grid import (
    DROP,
    PICKUP,
    REFUEL,
    E,
    GasGridEnv,
    GasGridState,
    S,
    gas_band,
    gas_depleted,
    manhattan_greedy_heuristic,
)


def test_is_an_environment():
    assert isinstance(GasGridEnv(), Environment)


def test_default_special_cells_2x2():
    env = GasGridEnv()
    assert env.refuel == (0, 1)
    assert env.dropoff == (1, 1)
    assert set(env.eligible_cells) == {(0, 0), (1, 0)}


def test_gas_band_quantization():
    assert gas_band(0, 4) == "critical"
    assert gas_band(4, 4) == "full"
    assert gas_band(2, 4) == "mid"
    assert gas_band(1, 4) == "low"


def test_initial_is_idle_p1_with_task_choices():
    env = GasGridEnv()
    s = env.initial_state()
    assert s.player is Player.P1 and s.task_loc is None
    assert s.gas == env.gas_max
    assert set(env.legal_actions(s)) == {("TASK", (0, 0)), ("TASK", (1, 0))}


def test_task_event_assigns_task_and_passes_to_p2():
    env = GasGridEnv()
    s = env.step(env.initial_state(), ("TASK", (1, 0)))
    assert s.task_loc == (1, 0)
    assert s.player is Player.P2


def test_active_p1_turn_reveals_observation_singleton():
    env = GasGridEnv()
    s = env.step(env.initial_state(), ("TASK", (1, 0)))
    s = env.step(s, S)  # P2 moves -> P1 active turn
    legal = env.legal_actions(s)
    assert legal == [env.observation(s)]  # forced observation reveal
    assert env.step(s, legal[0]).player is Player.P2


def test_full_reactive_delivery_succeeds():
    # 2x2: task at (1,0); deliver to dropoff (1,1) with refuel at (0,1).
    env = GasGridEnv()
    s = env.initial_state()
    s = env.step(s, ("TASK", (1, 0)))  # -> P2
    s = env.step(s, S)  # (0,0)->(1,0), gas 3 -> P1
    s = env.step(s, env.observation(s))  # reveal -> P2
    s = env.step(s, PICKUP)  # carrying -> P1
    s = env.step(s, env.observation(s))  # reveal -> P2
    s = env.step(s, E)  # (1,0)->(1,1) dropoff, gas 2 -> P1
    s = env.step(s, env.observation(s))  # reveal -> P2
    s = env.step(s, DROP)  # delivered
    assert env.is_terminal(s)
    assert s.delivered
    assert env.reward(s) == 1.0


def test_running_out_of_gas_is_terminal_failure():
    env = GasGridEnv(gas_max=1)
    s = env.step(env.initial_state(), ("TASK", (1, 0)))  # gas 1, P2
    s = env.step(s, S)  # move costs the last gas -> gas 0
    assert env.is_terminal(s)
    assert gas_depleted(s)
    assert env.reward(s) == -1.0


def test_timeout_is_terminal():
    env = GasGridEnv(max_steps=5)
    s = GasGridState(
        pos=(0, 0),
        gas=3,
        task_loc=(1, 0),
        carrying=False,
        delivered=False,
        player=Player.P2,
        step_count=5,
    )
    assert env.is_terminal(s)
    assert env.reward(s) == -1.0


def test_refuel_available_only_at_refuel_cell():
    env = GasGridEnv()
    at_refuel = GasGridState(
        pos=(0, 1),
        gas=2,
        task_loc=(1, 0),
        carrying=False,
        delivered=False,
        player=Player.P2,
        step_count=1,
    )
    assert REFUEL in env.legal_actions(at_refuel)
    assert env.step(at_refuel, REFUEL).gas == env.gas_max


def test_p1_alphabet_is_superset_of_legal():
    env = GasGridEnv()
    alphabet = set(env.p1_alphabet)
    s = env.step(env.initial_state(), ("TASK", (1, 0)))
    s = env.step(s, S)
    assert set(env.legal_actions(s)) <= alphabet  # reachable observation present
    assert ("TASK", (0, 0)) in alphabet


def test_heuristic_prefers_progress():
    env = GasGridEnv()
    carrying_near = GasGridState(
        pos=(1, 1),
        gas=3,
        task_loc=(1, 0),
        carrying=True,
        delivered=False,
        player=Player.P2,
        step_count=3,
    )
    not_carrying_far = GasGridState(
        pos=(0, 0),
        gas=3,
        task_loc=(1, 0),
        carrying=False,
        delivered=False,
        player=Player.P2,
        step_count=1,
    )
    assert manhattan_greedy_heuristic(env, carrying_near) > manhattan_greedy_heuristic(
        env, not_carrying_far
    )


def test_run_pals_learns_a_valid_controller():
    from pals.core.learner import run_pals
    from pals.core.preference import MinimaxPreferenceOracle

    env = GasGridEnv(gas_max=4, max_steps=10)
    oracle = MinimaxPreferenceOracle(env, manhattan_greedy_heuristic, depth=4)
    result = run_pals(env, oracle, depth_n=2, rollout_budget=20, rng=random.Random(0))
    assert len(result.model.states) >= 1
    for state in result.model.states:
        assert set(state.transitions) == set(env.p1_alphabet)
