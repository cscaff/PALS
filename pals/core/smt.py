"""Turn ordinal pairwise preferences into consistent numeric values via Z3.

The MCTS audit collects pairwise facts ("trace A is preferred to trace B") and
needs a single numeric value per trace consistent with all of them, so it can
rank and back-propagate them. This is a satisfiability problem over the reals:
assign each trace a value in ``[0, 1]`` such that every strict/equality
constraint holds. :class:`SMTValuer` wraps Z3 for exactly that, returning values
normalized to ``[0, 1]`` (least preferred → 0, most preferred → 1).
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import z3

from pals.core.preference import Preference
from pals.envs.base import Action

TraceKey = tuple[Action, ...]


class SMTValuer:
    """Assigns each seen trace a numeric value consistent with all preferences."""

    def __init__(self) -> None:
        self._solver = z3.Solver()
        self._vars: dict[TraceKey, z3.ArithRef] = {}
        self._last: dict[TraceKey, float] = {}

    def _var(self, trace: Sequence[Action]) -> z3.ArithRef:
        key = tuple(trace)
        var = self._vars.get(key)
        if var is None:
            var = z3.Real(f"v{len(self._vars)}")
            self._vars[key] = var
            self._solver.add(var >= 0, var <= 1)
        return var

    def add(
        self,
        trace1: Sequence[Action],
        trace2: Sequence[Action],
        preference: Preference,
    ) -> None:
        """Record a pairwise preference constraint."""
        v1, v2 = self._var(trace1), self._var(trace2)
        if preference is Preference.FIRST:
            self._solver.add(v1 > v2)
        elif preference is Preference.SECOND:
            self._solver.add(v2 > v1)
        else:
            self._solver.add(v1 == v2)

    def solve(self) -> dict[TraceKey, float] | None:
        """Solve for a normalized value per trace, or ``None`` if inconsistent."""
        if self._solver.check() != z3.sat:
            return None

        model = self._solver.model()
        raw: dict[TraceKey, float] = {}
        for key, var in self._vars.items():
            val = model[var]
            raw[key] = float(Fraction(str(val))) if val is not None else 0.0

        if not raw:
            self._last = {}
            return {}

        lo, hi = min(raw.values()), max(raw.values())
        if hi == lo:
            self._last = dict.fromkeys(raw, 0.5)
        else:
            self._last = {k: (v - lo) / (hi - lo) for k, v in raw.items()}
        return dict(self._last)

    def value(self, trace: Sequence[Action]) -> float | None:
        """Last solved value for ``trace`` (``None`` if unseen / unsolved)."""
        return self._last.get(tuple(trace))

    def is_satisfiable(self) -> bool:
        return self._solver.check() == z3.sat
