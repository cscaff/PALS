"""The System Under Learning that L* queries.

PALS learns the system player (P2). A membership query is a sequence of P1
inputs; the SUL simulates the strictly-alternating play (P1 input, then P2
response, then P1 input, ...) and returns P2's response after each P1 input.
P2's response is chosen by the preference oracle, *unless* a strategy override
is installed at that point.

Overrides are what make the SUL **mutable mid-run**: when the MCTS audit finds a
P2 deviation preferred over the current hypothesis, it calls ``update_strategy``
to pin that response, so subsequent queries reflect the improvement. The
discrepancy between the now-updated SUL and the old hypothesis is exactly the
counterexample that drives L* to refine. (Safety patches in a later phase reuse
the same override mechanism with a lock.)

Once a query runs off the tree (an illegal P1 input, or play continues past a
terminal state) every remaining output is the sink value ``None``. This keeps
the learned Mealy machine total and L*'s closedness trivially satisfied on those
rows.
"""

from __future__ import annotations

from collections.abc import Sequence

from pals.core.preference import PreferenceOracle
from pals.envs.base import Action, Environment, Player

# Interleaved trace ending at a P2-turn state (i.e. just after a P1 input).
P2NodeKey = tuple[Action, ...]


class PreferenceSUL:
    def __init__(self, env: Environment, oracle: PreferenceOracle) -> None:
        self.env = env
        self.oracle = oracle
        self._overrides: dict[P2NodeKey, Action] = {}
        self._locked: set[P2NodeKey] = set()
        self.num_queries = 0

    # ------------------------------------------------------------------
    # Membership query
    # ------------------------------------------------------------------

    def query(self, p1_inputs: Sequence[Action]) -> tuple[Action | None, ...]:
        """Return P2's response after each P1 input (``None`` past the tree)."""
        self.num_queries += 1
        env = self.env
        node = env.initial_state()
        trace: list[Action] = []
        outputs: list[Action | None] = []

        for p1 in p1_inputs:
            # Off-tree, terminal, or illegal P1 input -> sink for the rest.
            if (
                node is None
                or env.is_terminal(node)
                or p1 not in env.legal_actions(node)
            ):
                outputs.append(None)
                node = None
                continue

            trace.append(p1)
            node = env.step(node, p1)

            if env.is_terminal(node):
                # P1's move ended the game; P2 has no response.
                outputs.append(None)
                node = None
                continue

            p2 = self._p2_response(trace)
            outputs.append(p2)
            trace.append(p2)
            node = env.step(node, p2)

        return tuple(outputs)

    def _p2_response(self, trace_at_p2_node: Sequence[Action]) -> Action:
        key = tuple(trace_at_p2_node)
        if key in self._overrides:
            return self._overrides[key]
        return self.oracle.preferred_move(trace_at_p2_node)

    # ------------------------------------------------------------------
    # Mutation (MCTS audit; safety patches reuse this with a lock)
    # ------------------------------------------------------------------

    def update_strategy(
        self, trace_at_p2_node: Sequence[Action], response: Action
    ) -> bool:
        """Pin P2's response at ``trace_at_p2_node``. No-op (returns ``False``)
        if the site is locked (e.g. by a safety patch)."""
        key = tuple(trace_at_p2_node)
        if key in self._locked:
            return False
        self._overrides[key] = response
        return True

    def patch(self, trace_at_p2_node: Sequence[Action], response: Action) -> None:
        """Install an override and lock it against further ``update_strategy``."""
        key = tuple(trace_at_p2_node)
        self._overrides[key] = response
        self._locked.add(key)

    def current_response(self, trace_at_p2_node: Sequence[Action]) -> Action | None:
        """P2's response under the current strategy (override or oracle)."""
        state = self.env.get_node(trace_at_p2_node)
        if (
            state is None
            or self.env.is_terminal(state)
            or self.env.current_player(state) is not Player.P2
        ):
            return None
        return self._p2_response(trace_at_p2_node)

    @staticmethod
    def p1_inputs_of(trace: Sequence[Action]) -> list[Action]:
        """Extract P1 inputs (even indices) from an interleaved trace."""
        return [trace[i] for i in range(0, len(trace), 2)]
