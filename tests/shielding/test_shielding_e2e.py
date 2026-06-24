"""The active-shielding experiment + focused shield/SUL integration tests.

The headline result the reviewers asked for: an unshielded controller violates a
safety spec, and the shield makes it safe — on the gas-grid where preference
(deliver fast, gas-blind) is deliberately *misaligned* with safety (G(gas>0)).
"""

import random

from pals.bench.evaluate import play_game
from pals.bench.players import PALSPlayer, RandomPlayer, greedy_action
from pals.core.learner import run_pals
from pals.core.preference import MinimaxPreferenceOracle
from pals.core.sul import PreferenceSUL
from pals.envs.gas_grid import (
    REFUEL,
    GasGridEnv,
    gas_depleted,
    manhattan_greedy_heuristic,
    safety_state_key,
)
from pals.shielding.model_check import find_violation
from pals.shielding.shield_oracle import ShieldOracle
from pals.shielding.spec import SafetySpec


def _corridor():
    """1x3 corridor where the gas-blind greedy policy runs the tank to zero."""
    return GasGridEnv(
        rows=1,
        cols=3,
        home=(0, 0),
        refuel=(0, 1),
        dropoff=(0, 2),
        gas_max=2,
        eligible_cells=((0, 0),),
        gas_is_fatal=False,
    )


def _oracle(env):
    return MinimaxPreferenceOracle(env, manhattan_greedy_heuristic, depth=4)


def _spec():
    return SafetySpec(gas_depleted, name="G(gas>0)", state_key=safety_state_key)


def test_unshielded_controller_violates_the_safety_spec():
    env = _corridor()
    result = run_pals(
        env,
        _oracle(env),
        depth_n=2,
        rollout_budget=10,
        use_pac=False,
        rng=random.Random(0),
    )
    # The gas-blind preferred policy drives the tank to empty.
    assert find_violation(env, result.model, gas_depleted) is not None
    assert result.shield_patches == 0


def test_shielded_controller_is_safe():
    env = _corridor()
    result = run_pals(
        env,
        _oracle(env),
        depth_n=2,
        rollout_budget=10,
        use_pac=False,
        spec=_spec(),
        rng=random.Random(0),
    )
    # With the shield, no environment behaviour can drive gas to zero.
    assert find_violation(env, result.model, gas_depleted) is None
    assert result.shield_patches >= 1


def test_shielded_controller_stays_safe_and_still_delivers():
    # Passing the preference as a tie-breaker (prefer_action) makes the shield
    # keep the controller's move wherever it is safe and refuel only where
    # needed -- so the controller is both safe AND completes the task.
    env = _corridor()
    result = run_pals(
        env,
        _oracle(env),
        depth_n=2,
        rollout_budget=10,
        use_pac=False,
        spec=_spec(),
        prefer_action=greedy_action(env, manhattan_greedy_heuristic),
        rng=random.Random(0),
    )
    assert find_violation(env, result.model, gas_depleted) is None  # safe
    assert result.shield_patches == 1  # one well-placed refuel patch
    reward = play_game(env, RandomPlayer(), PALSPlayer(result.model), random.Random(0))
    assert reward == 1.0  # delivered (recovers the objective the spec permits)


def test_shield_oracle_patches_unsafe_then_converges():
    env = _corridor()
    # First learn an unsafe controller without the shield.
    unsafe = run_pals(
        env,
        _oracle(env),
        depth_n=2,
        rollout_budget=10,
        use_pac=False,
        rng=random.Random(0),
    )
    sul = unsafe.sul
    shield = ShieldOracle(env, sul, _spec())
    cex = shield.find_counterexample(unsafe.model)
    assert cex is not None  # violation found -> CEX to refine on
    assert shield.patches_installed >= 1


def test_safety_patch_beats_mcts_override():
    from pals.envs.base import Player

    env = _corridor()
    sul = PreferenceSUL(env, _oracle(env))
    sul.state_key_fn = safety_state_key

    # Boundary P2 node: at (0,1) carrying one gas, the safe action is REFUEL.
    boundary_key = ((0, 1), 1, (0, 0), True, Player.P2)
    sul.patch_state(boundary_key, REFUEL)

    # Interleaved trace ending at that P2 node (obs reveals are forced inputs).
    trace = [
        ("TASK", (0, 0)),
        "PICKUP",
        ((0, 0), "full", (0, 0), True),
        "E",
        ((0, 1), "mid", (0, 0), True),
    ]
    assert sul.current_response(trace) == REFUEL  # state-keyed safety patch wins
    # MCTS cannot override a safety-locked state.
    assert sul.update_strategy(trace, "W") is False
    assert sul.current_response(trace) == REFUEL
