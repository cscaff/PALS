"""The :class:`Environment` abstraction every PALS environment implements.

PALS frames every problem as a two-player, turn-based environment:

* **P1 — the environment.** Supplies inputs. P1 is *enumerated, not optimized*:
  every legal P1 input is treated as equally valid. P1's moves form the L*
  input alphabet.
* **P2 — the system.** Produces outputs in response. P2's strategy is exactly
  what PALS learns (as a symbolic Mealy machine).

A subclass implements six primitives over **opaque hashable states**
(``initial_state``, ``current_player``, ``legal_actions``, ``step``,
``is_terminal``, ``reward``). States are hashable so oracles can memoize over
them. The trace-navigation helpers (``get_node`` and friends) are derived
generically here, so the algorithm can drive any environment uniformly without
the subclass re-implementing them.

A *trace* is the sequence of actions taken from the initial state, interleaving
P1 and P2 moves in turn order: ``[a_P1, a_P2, a_P1, ...]``. Turn order is
governed by ``current_player`` — strict alternation is the common case but is
not assumed, so environments where one player moves several times in a row are
also expressible.

This interface is deliberately Gymnasium-friendly: a state can be an opaque
encoded observation and transitions are computed on demand, so adapters need
not materialize the whole game tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from enum import Enum

# A state or action is any hashable value. Concrete environments pick the
# representation (tuples, ints, NamedTuples, ...); the algorithm treats them
# opaquely and only relies on hashability and equality.
State = Hashable
Action = Hashable


class Player(str, Enum):
    """The two roles. ``str`` mixin keeps values printable and trace-friendly."""

    P1 = "P1"  # environment — supplies inputs (the L* input alphabet)
    P2 = "P2"  # system — the controller/strategy PALS learns


class Environment(ABC):
    """Two-player turn-based environment as a (possibly infinite) game tree."""

    # ------------------------------------------------------------------
    # Primitives — subclasses must implement these six.
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def p1_alphabet(self) -> list[Action]:
        """The full set of P1 input symbols — i.e. the L* input alphabet.

        Must be a superset of ``legal_actions(s)`` for every P1-turn state ``s``
        reachable in the environment.
        """

    @abstractmethod
    def initial_state(self) -> State:
        """The root state where every trace begins."""

    @abstractmethod
    def current_player(self, state: State) -> Player:
        """Whose turn it is at ``state``. Only meaningful for non-terminal states."""

    @abstractmethod
    def legal_actions(self, state: State) -> list[Action]:
        """Actions available to the player to move at ``state`` (``[]`` if terminal)."""

    @abstractmethod
    def step(self, state: State, action: Action) -> State:
        """The state reached by taking ``action`` at ``state``.

        Behaviour is only defined for ``action in legal_actions(state)``.
        """

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Whether ``state`` is a leaf (the game has ended)."""

    @abstractmethod
    def reward(self, state: State) -> float:
        """Terminal payoff at ``state``, from P2's (the system's) perspective.

        Only defined at terminal states; callers gate on ``is_terminal`` first.
        """

    # ------------------------------------------------------------------
    # Derived trace navigation — generic, shared by every environment.
    # ------------------------------------------------------------------

    def get_node(self, trace: Sequence[Action]) -> State | None:
        """Fold ``trace`` from the initial state to the state it reaches.

        Returns ``None`` if the trace runs off the tree — i.e. it takes an
        action that is not legal at the current state, or continues past a
        terminal state.
        """
        state = self.initial_state()
        for action in trace:
            if self.is_terminal(state):
                return None
            if action not in self.legal_actions(state):
                return None
            state = self.step(state, action)
        return state

    def current_player_at(self, trace: Sequence[Action]) -> Player | None:
        """``current_player`` of the state ``trace`` reaches, or ``None`` if
        the trace is invalid or terminal."""
        state = self.get_node(trace)
        if state is None or self.is_terminal(state):
            return None
        return self.current_player(state)

    def legal_actions_at(self, trace: Sequence[Action]) -> list[Action]:
        """Legal actions at the state ``trace`` reaches (``[]`` if invalid/terminal)."""
        state = self.get_node(trace)
        if state is None or self.is_terminal(state):
            return []
        return self.legal_actions(state)

    def p1_legal_inputs(self, trace: Sequence[Action]) -> list[Action]:
        """Legal actions at ``trace`` when it is P1's turn there (else ``[]``)."""
        if self.current_player_at(trace) is not Player.P1:
            return []
        return self.legal_actions_at(trace)

    def p2_legal_moves(self, trace: Sequence[Action]) -> list[Action]:
        """Legal actions at ``trace`` when it is P2's turn there (else ``[]``)."""
        if self.current_player_at(trace) is not Player.P2:
            return []
        return self.legal_actions_at(trace)
