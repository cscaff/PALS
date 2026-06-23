"""Generic two-player safety-game solver (pure Python).

Given any :class:`~pals.envs.base.Environment` and a ``is_bad`` predicate marking
spec-violating states (e.g. ``gas <= 0`` for ``G(gas > 0)``), compute:

* the **winning set** ``W`` — states from which the controller (P2) can keep the
  play safe forever no matter what the environment (P1) does, and
* a **safe strategy** — one safe action per winning P2 state.

This is the standard backward attractor over the reachable state graph:

* a **P1 (environment)** state is losing if *any* successor is losing (adversarial);
* a **P2 (controller)** state is losing if *all* successors are losing (cooperative);
* bad states are losing outright.

No external model checker is required — this is the pure-Python default backend
for safety specs. An optional ``state_key`` quotients the state space (the caller
must ensure it is a sound abstraction for the spec, e.g. dropping a step counter
that the spec ignores); by default the raw (hashable) state is the key, which is
exact whenever the reachable state space is finite.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from pals.envs.base import Action, Environment, Player, State

StateKey = object


@dataclass
class SafetyGame:
    winning: set[StateKey]  # keys of states from which P2 can stay safe
    strategy: dict[StateKey, Action]  # one safe action per winning P2 state
    states: dict[StateKey, State]  # a representative state per key

    def safe_action(self, state_key: StateKey) -> Action | None:
        return self.strategy.get(state_key)


def solve_safety_game(
    env: Environment,
    is_bad: Callable[[State], bool],
    prefer_action: Callable[[State], Action | None] | None = None,
    state_key: Callable[[State], StateKey] | None = None,
) -> SafetyGame:
    key = state_key or (lambda s: s)

    # ----- reachable state graph -----
    states: dict[StateKey, State] = {}
    transitions: dict[StateKey, dict[Action, StateKey]] = {}
    owner: dict[StateKey, Player | None] = {}
    bad: set[StateKey] = set()

    queue: deque[State] = deque([env.initial_state()])
    while queue:
        s = queue.popleft()
        k = key(s)
        if k in states:
            continue
        states[k] = s
        if is_bad(s):
            bad.add(k)
        if env.is_terminal(s):
            transitions[k] = {}
            owner[k] = None
            continue
        owner[k] = env.current_player(s)
        edges: dict[Action, StateKey] = {}
        for a in env.legal_actions(s):
            child = env.step(s, a)
            edges[a] = key(child)
            queue.append(child)
        transitions[k] = edges

    # ----- backward attractor onto the bad set -----
    predecessors: dict[StateKey, list[StateKey]] = {k: [] for k in states}
    for k, edges in transitions.items():
        for ck in edges.values():
            predecessors[ck].append(k)

    losing = set(bad)
    # Remaining successor edges not yet known to be losing. Counts *all* edges
    # (with multiplicity); the worklist below decrements one per losing edge,
    # including the initially-bad ones — so do not pre-filter here.
    remaining = {
        k: len(transitions[k])
        for k in states
        if owner[k] is Player.P2 and transitions[k]
    }

    worklist: deque[StateKey] = deque(losing)
    for k, count in remaining.items():
        if count == 0 and k not in losing:
            losing.add(k)
            worklist.append(k)

    while worklist:
        lost = worklist.popleft()
        for pred in predecessors[lost]:
            if pred in losing:
                continue
            if owner[pred] is Player.P1:
                losing.add(pred)  # adversarial: one bad successor is enough
                worklist.append(pred)
            elif owner[pred] is Player.P2:
                remaining[pred] -= 1
                if remaining[pred] <= 0:  # all successors losing
                    losing.add(pred)
                    worklist.append(pred)

    winning = set(states) - losing

    # ----- extract one safe action per winning P2 state -----
    strategy: dict[StateKey, Action] = {}
    for k in winning:
        if owner[k] is not Player.P2:
            continue
        safe = [a for a, ck in transitions[k].items() if ck in winning]
        if not safe:
            continue
        chosen = None
        if prefer_action is not None:
            pref = prefer_action(states[k])
            if pref in safe:
                chosen = pref
        strategy[k] = chosen if chosen is not None else safe[0]

    return SafetyGame(winning=winning, strategy=strategy, states=states)
