"""Mealy-machine L* over a (possibly mutating) :class:`PreferenceSUL`.

Classic Angluin L* adapted two ways:

1. **Mealy outputs.** Membership queries return P2's output after each P1 input,
   so rows are tuples of output-tuples and the hypothesis is a Mealy machine.
2. **A mutable target.** The SUL changes mid-run as the MCTS audit installs
   strategy overrides. Rather than the old repo's fragile targeted-invalidation,
   we simply **re-query the whole table each round**: queries are deterministic
   and cheap, so a fresh fill is always consistent with the current SUL and
   sidesteps a class of staleness bugs. (Optimize later if a benchmark needs it.)

Observation table: ``S`` are P1-input prefixes (with the empty prefix), ``E`` are
P1-input suffixes (seeded with single inputs), and ``T[(s, e)]`` is the output
tuple for the ``e`` portion of querying ``s + e``. Two prefixes are equivalent
when their rows agree on every suffix.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pals.core.sul import PreferenceSUL
from pals.envs.base import Action

Output = Action | None
Row = tuple[tuple[Output, ...], ...]


class EquivalenceOracle(Protocol):
    """Returns a P1-input counterexample where hypothesis and SUL differ, or
    ``None`` if equivalent. May mutate the SUL before returning."""

    def find_counterexample(self, hypothesis: MealyMachine) -> list[Action] | None: ...


class MealyState:
    def __init__(self, state_id: str) -> None:
        self.state_id = state_id
        self.transitions: dict[Action, tuple[Output, MealyState]] = {}

    def __repr__(self) -> str:
        return f"MealyState({self.state_id})"


class MealyMachine:
    def __init__(self, states: list[MealyState], initial: MealyState) -> None:
        self.states = states
        self.initial_state = initial
        self.current_state = initial

    def reset_to_initial(self) -> None:
        self.current_state = self.initial_state

    def step(self, inp: Action) -> Output:
        """Advance on ``inp``; return the emitted output (``None`` if no edge)."""
        edge = self.current_state.transitions.get(inp)
        if edge is None:
            return None
        output, nxt = edge
        self.current_state = nxt
        return output

    def output_of(self, p1_inputs: Sequence[Action]) -> list[Output]:
        """Run ``p1_inputs`` from the initial state, collecting outputs."""
        self.reset_to_initial()
        return [self.step(a) for a in p1_inputs]


class MealyLStar:
    def __init__(
        self,
        alphabet: Sequence[Action],
        sul: PreferenceSUL,
        eq_oracle: EquivalenceOracle,
        verbose: bool = False,
        max_rounds: int = 1000,
    ) -> None:
        self.alphabet = list(alphabet)
        self.sul = sul
        self.eq_oracle = eq_oracle
        self.verbose = verbose
        self.max_rounds = max_rounds

        self.S: list[tuple[Action, ...]] = [()]
        self.E: list[tuple[Action, ...]] = [(a,) for a in self.alphabet]
        self.T: dict[tuple[tuple, tuple], tuple[Output, ...]] = {}

    # ------------------------------------------------------------------
    # Table filling (always fresh — the SUL may have mutated)
    # ------------------------------------------------------------------

    def _fill(self) -> None:
        rows = set(self.S) | {s + (a,) for s in self.S for a in self.alphabet}
        for s in rows:
            for e in self.E:
                outputs = self.sul.query(s + e)
                self.T[(s, e)] = outputs[len(s) :]

    def _row(self, s: tuple[Action, ...]) -> Row:
        return tuple(self.T[(s, e)] for e in self.E)

    def _known_rows(self) -> set[Row]:
        return {self._row(s) for s in self.S}

    # ------------------------------------------------------------------
    # Closedness & consistency
    # ------------------------------------------------------------------

    def _close(self) -> bool:
        known = self._known_rows()
        for s in self.S:
            for a in self.alphabet:
                sa = s + (a,)
                if self._row(sa) not in known:
                    self.S.append(sa)
                    self._fill()
                    return True
        return False

    def _make_consistent(self) -> bool:
        for i, s1 in enumerate(self.S):
            for s2 in self.S[i + 1 :]:
                if self._row(s1) != self._row(s2):
                    continue
                for a in self.alphabet:
                    for e in self.E:
                        if self.T[(s1 + (a,), e)] != self.T[(s2 + (a,), e)]:
                            new_e = (a,) + e
                            if new_e not in self.E:
                                self.E.append(new_e)
                                self._fill()
                                return True
        return False

    # ------------------------------------------------------------------
    # Hypothesis construction
    # ------------------------------------------------------------------

    def _build(self) -> MealyMachine:
        row_to_state: dict[Row, MealyState] = {}
        for s in self.S:
            r = self._row(s)
            if r not in row_to_state:
                row_to_state[r] = MealyState(f"q{len(row_to_state)}")

        for s in self.S:
            src = row_to_state[self._row(s)]
            for a in self.alphabet:
                output = self.T[(s, (a,))][0]
                dst = row_to_state[self._row(s + (a,))]
                src.transitions[a] = (output, dst)

        machine = MealyMachine(list(row_to_state.values()), row_to_state[self._row(())])
        if self.verbose:
            print(f"  hypothesis: {len(machine.states)} states")
        return machine

    def _add_counterexample(self, cex: Sequence[Action]) -> None:
        for i in range(1, len(cex) + 1):
            prefix = tuple(cex[:i])
            if prefix not in self.S:
                self.S.append(prefix)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> MealyMachine:
        self._fill()
        round_num = 0
        while True:
            while self._close() or self._make_consistent():
                pass

            hypothesis = self._build()
            round_num += 1
            if self.verbose:
                print(f"round {round_num}: {len(hypothesis.states)} states")

            cex = self.eq_oracle.find_counterexample(hypothesis)
            if cex is None:
                return hypothesis

            self._add_counterexample(cex)
            self._fill()  # SUL may have mutated; refresh the whole table

            if round_num >= self.max_rounds:
                raise RuntimeError(
                    f"L* did not converge within {self.max_rounds} rounds"
                )
