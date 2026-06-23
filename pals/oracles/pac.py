"""PAC equivalence oracle (Angluin/Kearns-Vazirani sample schedule).

Samples random legal walks of the environment and tests the hypothesis against
the SUL on each. On the ``i``-th query it draws
``m_i = ceil((1/eps)(ln(1/delta) + i*ln 2))`` samples; if none disagree, then
with probability >= ``1 - delta`` the hypothesis differs from the SUL on at most
an ``eps`` fraction of the sampling distribution. Composed *after* the MCTS audit
(which mutates the SUL), it validates that the final hypothesis matches the
now-stable SUL.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from pals.core.lstar import MealyMachine
from pals.core.sul import PreferenceSUL
from pals.envs.base import Action, Environment, Player


class PACEquivalenceOracle:
    def __init__(
        self,
        env: Environment,
        sul: PreferenceSUL,
        eps: float = 0.05,
        delta: float = 0.05,
        max_walk: int = 20,
        rng: random.Random | None = None,
    ) -> None:
        self.env = env
        self.sul = sul
        self.eps = eps
        self.delta = delta
        self.max_walk = max_walk
        self.rng = rng or random.Random()
        self.num_queries = 0

    def _sample_p1_walk(self) -> list[Action]:
        """Random legal walk from the root, collecting only P1 inputs."""
        env = self.env
        state = env.initial_state()
        p1_inputs: list[Action] = []
        while (
            state is not None
            and not env.is_terminal(state)
            and len(p1_inputs) < self.max_walk
        ):
            actions = env.legal_actions(state)
            if not actions:
                break
            action = self.rng.choice(actions)
            if env.current_player(state) is Player.P1:
                p1_inputs.append(action)
            state = env.step(state, action)
        return p1_inputs

    def find_counterexample(self, hypothesis: MealyMachine) -> list[Action] | None:
        self.num_queries += 1
        m = math.ceil(
            (1 / self.eps) * (math.log(1 / self.delta) + self.num_queries * math.log(2))
        )
        for _ in range(m):
            seq = self._sample_p1_walk()
            if not seq:
                continue
            cex = self._first_disagreement(hypothesis, seq)
            if cex is not None:
                return cex
        return None

    def _first_disagreement(
        self, hypothesis: MealyMachine, seq: Sequence[Action]
    ) -> list[Action] | None:
        hyp_out = hypothesis.output_of(seq)
        sul_out = self.sul.query(seq)
        for k in range(len(seq)):
            if hyp_out[k] != sul_out[k]:
                return list(seq[: k + 1])
        return None
