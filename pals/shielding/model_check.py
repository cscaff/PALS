"""Pure-Python safety model checking (the default backend).

Checks whether a learned controller can reach a spec-violating state. The
controller is a Mealy machine over P1 inputs; the environment supplies P1 inputs
(adversarially — we branch over *all* legal ones) and the controller answers with
its Mealy output. We BFS the product (Mealy state x env state) and return the
shortest interleaved action trace to a bad state, or ``None`` if the controller
is safe against every environment behaviour.

This is the pure-Python default; a Spot/LTL backend can be slotted in later
behind the same ``find_violation`` signature.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from pals.core.lstar import MealyMachine, MealyState
from pals.envs.base import Action, Environment, Player, State


def find_violation(
    env: Environment,
    hypothesis: MealyMachine,
    is_bad: Callable[[State], bool],
) -> list[Action] | None:
    """Shortest interleaved trace to a bad state under ``hypothesis``, or None."""
    start_env = env.initial_state()
    start_m = hypothesis.initial_state
    if is_bad(start_env):
        return []

    # Frontier holds (mealy_state, env_state-at-a-P1-turn, trace-so-far).
    frontier: deque[tuple[MealyState, State, list[Action]]] = deque(
        [(start_m, start_env, [])]
    )
    visited: set[tuple[int, State]] = {(id(start_m), start_env)}

    while frontier:
        m_state, env_state, trace = frontier.popleft()
        if env.is_terminal(env_state) or env.current_player(env_state) is not Player.P1:
            continue

        for p1 in env.legal_actions(env_state):
            edge = m_state.transitions.get(p1)
            if edge is None:
                continue  # controller has no response to this input
            output, next_m = edge
            after_p1 = env.step(env_state, p1)

            if env.is_terminal(after_p1):
                if is_bad(after_p1):
                    return [*trace, p1]
                continue
            if output not in env.legal_actions(after_p1):
                continue  # controller's action is not legal here; off-path

            after_p2 = env.step(after_p1, output)
            new_trace = [*trace, p1, output]
            if is_bad(after_p2):
                return new_trace

            visit_key = (id(next_m), after_p2)
            if visit_key not in visited:
                visited.add(visit_key)
                frontier.append((next_m, after_p2, new_trace))

    return None
