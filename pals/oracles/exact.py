"""Bounded exact equivalence oracle.

Enumerates every P1-input sequence up to a bounded length (shortest first) and
returns the first one on which the hypothesis and the SUL disagree. Exhaustive
within the bound, so when the preference oracle is exact this recovers a Mealy
machine provably equivalent to the SUL over all inputs of that length. Mainly a
ground-truth oracle for tests and the exact-oracle regime; the MCTS/PAC oracles
handle the realistic suboptimal-oracle setting.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from pals.core.lstar import MealyMachine
from pals.core.sul import PreferenceSUL
from pals.envs.base import Action


class BoundedExactOracle:
    def __init__(
        self, alphabet: Sequence[Action], sul: PreferenceSUL, max_length: int
    ) -> None:
        self.alphabet = list(alphabet)
        self.sul = sul
        self.max_length = max_length

    def find_counterexample(self, hypothesis: MealyMachine) -> list[Action] | None:
        for length in range(1, self.max_length + 1):
            for seq in itertools.product(self.alphabet, repeat=length):
                if hypothesis.output_of(seq) != list(self.sul.query(seq)):
                    return list(seq)
        return None
