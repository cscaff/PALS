"""Unit tests for the pure-Python safety model checker."""

from _graph_env import GraphEnv

from pals.core.lstar import MealyMachine, MealyState
from pals.envs.base import Player
from pals.shielding.model_check import find_violation


def _single_state_mealy(transitions: dict):
    q = MealyState("q0")
    for inp, out in transitions.items():
        q.transitions[inp] = (out, q)
    return MealyMachine([q], q)


def _alternating_env(p2_target):
    # 0 (P1) --in--> 1 (P2) --act--> p2_target
    return GraphEnv(
        start=0,
        players={0: Player.P1, 1: Player.P2, 2: Player.P2, 3: Player.P2},
        edges={0: {"in": 1}, 1: {"act": p2_target}},
        terminals={2, 3},
    )


def test_finds_violation_when_controller_reaches_bad():
    env = _alternating_env(p2_target=2)  # act leads to state 2
    mealy = _single_state_mealy({"in": "act"})
    trace = find_violation(env, mealy, is_bad=lambda s: s == 2)
    assert trace == ["in", "act"]


def test_no_violation_when_controller_stays_safe():
    env = _alternating_env(p2_target=3)  # act leads to safe terminal 3
    mealy = _single_state_mealy({"in": "act"})
    assert find_violation(env, mealy, is_bad=lambda s: s == 2) is None


def test_bad_initial_state_is_immediate_violation():
    env = _alternating_env(p2_target=3)
    mealy = _single_state_mealy({"in": "act"})
    assert find_violation(env, mealy, is_bad=lambda s: s == 0) == []
