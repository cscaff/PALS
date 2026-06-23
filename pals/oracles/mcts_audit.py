"""MCTS audit equivalence oracle — the heart of PALS.

A perfect preference oracle would make L* alone sufficient. Real oracles are only
*locally* optimal, so after L* settles on a hypothesis ``H`` this stage audits it
for greedy suboptimality. Per round it runs up to ``K`` rollouts; each rollout:

1. **Sample a deviation point** ``τσ`` — a prefix of a play of ``H`` ending at a
   P2 turn, biased toward under-explored points (Table B).
2. **Roll out two futures to depth ``N``:** ``S_H`` follows ``H``'s output at
   ``τσ``; ``S_dev`` takes a *different* P2 output (UCB-guided via Table B) and
   continues.
3. **Vote.** Compare every ``S_dev`` leaf against every ``S_H`` leaf under the
   preference oracle; the deviation wins on a strict majority.
4. **Value & back-propagate.** An SMT valuation of the leaves feeds Table B (so
   future rollouts are better guided), with visit counts incremented.
5. If the deviation wins, **mutate the SUL** to emit the deviating output at
   ``τσ`` and return the P1-input prefix as a counterexample, so L* refines.

After a full round of ``K`` rollouts finds no improving deviation, the hypothesis
is locally optimal under the oracle and the audit reports convergence (and stays
converged, letting later stages finish).
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from pals.core.lstar import MealyMachine
from pals.core.preference import Preference, PreferenceOracle
from pals.core.smt import SMTValuer
from pals.core.sul import PreferenceSUL
from pals.core.table_b import TableB
from pals.envs.base import Action, Environment, Player

Trace = list[Action]


class MCTSAuditOracle:
    def __init__(
        self,
        env: Environment,
        sul: PreferenceSUL,
        oracle: PreferenceOracle,
        table_b: TableB,
        depth_n: int,
        rollout_budget: int = 200,
        max_subtrace: int = 20,
        temperature: float = 1.0,
        rng: random.Random | None = None,
        verbose: bool = False,
    ) -> None:
        self.env = env
        self.sul = sul
        self.oracle = oracle
        self.table_b = table_b
        self.depth_n = depth_n
        self.K = rollout_budget
        self.max_subtrace = max_subtrace
        self.temperature = temperature
        self.rng = rng or random.Random()
        self.verbose = verbose

        self.hypothesis: MealyMachine | None = None
        self.num_queries = 0
        self.accepted_deviations = 0
        self._converged = False

    # ------------------------------------------------------------------
    # Equivalence-oracle entry point
    # ------------------------------------------------------------------

    def find_counterexample(self, hypothesis: MealyMachine) -> list[Action] | None:
        self.num_queries += 1
        if self._converged:
            return None
        self.hypothesis = hypothesis

        for _ in range(self.K):
            subtrace = self._sample_deviation_point()
            if subtrace is None:
                continue
            deviation = self._choose_deviation(subtrace)
            if deviation is None:
                continue
            dev_leaves = self._rollout_deviation(subtrace, deviation)
            hyp_leaves = self._rollout_hypothesis(subtrace)
            values, dev_preferred = self._vote_and_value(dev_leaves, hyp_leaves)
            self._backpropagate(subtrace, values)

            if dev_preferred and self.sul.update_strategy(subtrace, deviation):
                self.accepted_deviations += 1
                if self.verbose:
                    print(f"  [mcts] accepted deviation at {subtrace} -> {deviation}")
                return self.sul.p1_inputs_of(subtrace)

        self._converged = True
        return None

    # ------------------------------------------------------------------
    # 1. Sample a deviation point (a P2-turn prefix of a play of H)
    # ------------------------------------------------------------------

    def _sample_deviation_point(self) -> Trace | None:
        assert self.hypothesis is not None
        self.hypothesis.reset_to_initial()
        env = self.env
        state = env.initial_state()
        trace: Trace = []
        candidates: list[Trace] = []

        while not env.is_terminal(state) and len(trace) < self.max_subtrace:
            if env.current_player(state) is not Player.P1:
                break
            p1 = self.rng.choice(env.legal_actions(state))
            trace.append(p1)
            state = env.step(state, p1)
            lam_h = self.hypothesis.step(p1)

            if env.is_terminal(state) or lam_h is None:
                break  # P1's move ended the game; no P2 deviation here
            candidates.append(list(trace))  # prefix ends at this P2 turn

            trace.append(lam_h)
            state = env.step(state, lam_h)

        if not candidates:
            return None
        # Bias toward under-explored deviation points (fewer Table B visits).
        weights = [1.0 / (1 + self._visits_at(c)) for c in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _visits_at(self, p2_node: Sequence[Action]) -> int:
        return sum(s.visits for s in self.table_b.actions_at(p2_node).values())

    # ------------------------------------------------------------------
    # 2. Choose a deviating P2 output (≠ H's) at the deviation point
    # ------------------------------------------------------------------

    def _choose_deviation(self, subtrace: Trace) -> Action | None:
        lam_h = self._hyp_output(self.sul.p1_inputs_of(subtrace))
        alternatives = [m for m in self.env.p2_legal_moves(subtrace) if m != lam_h]
        if not alternatives:
            return None
        for m in alternatives:
            self.table_b.record_visit(subtrace, m)
        return self.table_b.sample_action(
            subtrace, alternatives, temperature=self.temperature, rng=self.rng
        )

    # ------------------------------------------------------------------
    # 3. Roll out the deviation (Table B guided) and the hypothesis
    # ------------------------------------------------------------------

    def _rollout_deviation(self, subtrace: Trace, deviation: Action) -> list[Trace]:
        frontier: list[Trace] = [subtrace + [deviation]]
        completed: list[Trace] = []
        for _ in range(self.depth_n - 1):
            nxt: list[Trace] = []
            for trace in frontier:
                state = self.env.get_node(trace)
                if state is None or self.env.is_terminal(state):
                    completed.append(trace)
                    continue
                for p1 in self.env.legal_actions(state):  # P1 turn: branch
                    after_p1 = trace + [p1]
                    p1_state = self.env.step(state, p1)
                    if self.env.is_terminal(p1_state):
                        completed.append(after_p1)
                        continue
                    moves = self.env.legal_actions(p1_state)
                    for m in moves:
                        self.table_b.record_visit(after_p1, m)
                    p2 = self.table_b.sample_action(
                        after_p1, moves, temperature=self.temperature, rng=self.rng
                    )
                    nxt.append(after_p1 + [p2])
            frontier = nxt
            if not frontier:
                break
        return completed + frontier

    def _rollout_hypothesis(self, subtrace: Trace) -> list[Trace]:
        lam_h = self._hyp_output(self.sul.p1_inputs_of(subtrace))
        frontier: list[Trace] = [subtrace + [lam_h]]
        completed: list[Trace] = []
        for _ in range(self.depth_n - 1):
            nxt: list[Trace] = []
            for trace in frontier:
                state = self.env.get_node(trace)
                if state is None or self.env.is_terminal(state):
                    completed.append(trace)
                    continue
                for p1 in self.env.legal_actions(state):
                    after_p1 = trace + [p1]
                    p1_state = self.env.step(state, p1)
                    if self.env.is_terminal(p1_state):
                        completed.append(after_p1)
                        continue
                    p2 = self._hyp_output(self.sul.p1_inputs_of(after_p1))
                    if p2 is None or p2 not in self.env.legal_actions(p1_state):
                        completed.append(after_p1)
                        continue
                    nxt.append(after_p1 + [p2])
            frontier = nxt
            if not frontier:
                break
        return completed + frontier

    def _hyp_output(self, p1_inputs: Sequence[Action]) -> Action | None:
        assert self.hypothesis is not None
        outputs = self.hypothesis.output_of(p1_inputs)
        return outputs[-1] if outputs else None

    # ------------------------------------------------------------------
    # 4. Vote + SMT valuation, then back-propagate values into Table B
    # ------------------------------------------------------------------

    def _vote_and_value(
        self, dev_leaves: list[Trace], hyp_leaves: list[Trace]
    ) -> tuple[dict, bool]:
        smt = SMTValuer()
        dev_wins = 0
        total = 0
        for ce in dev_leaves:
            for he in hyp_leaves:
                pref = self.oracle.compare(ce, he)
                smt.add(ce, he, pref)
                if pref is Preference.FIRST:
                    dev_wins += 1
                total += 1
        values = smt.solve() or {}
        dev_preferred = total > 0 and dev_wins / total > 0.5
        return values, dev_preferred

    def _backpropagate(self, subtrace: Trace, values: dict) -> None:
        # Seed leaf values, then average upward one P1/P2 round at a time until
        # the deviation point, so Table B's UCB reflects what was learned.
        level: list[tuple] = []
        for trace_key, value in values.items():
            trace = list(trace_key)
            if len(trace) >= 2:
                self.table_b.update_value(trace[:-1], trace[-1], value)
                level.append(tuple(trace))

        base = len(subtrace)
        while level:
            parents = {tuple(t[:-2]) for t in level if len(t) - 2 >= base}
            if not parents:
                break
            for parent in parents:
                children = self.table_b.actions_at(parent)
                if not children:
                    continue
                avg = sum(s.value for s in children.values()) / len(children)
                if len(parent) >= 2:
                    self.table_b.update_value(parent[:-1], parent[-1], avg)
            level = list(parents)
