"""Composite equivalence oracle — a chain of stages.

On each ``find_counterexample`` call, stages run in order and the first
non-``None`` counterexample is returned. A typical chain is
``[MCTSAudit, PAC, Shield]``: the audit drives strategy convergence (mutating the
SUL), PAC validates behavioural equivalence against the stabilized SUL, and the
shield enforces safety. Joint convergence (every stage returns ``None``) ends L*.

Because each PALS stage is just a stage here, **ablations are config, not
forks**: drop the audit, the shield, or PAC by leaving it out of the chain.
"""

from __future__ import annotations

from pals.core.lstar import EquivalenceOracle, MealyMachine
from pals.envs.base import Action


class CompositeEquivalenceOracle:
    def __init__(self, *stages: EquivalenceOracle, verbose: bool = False) -> None:
        if not stages:
            raise ValueError("CompositeEquivalenceOracle needs at least one stage")
        self.stages = stages
        self.verbose = verbose
        self.num_queries = 0

    def find_counterexample(self, hypothesis: MealyMachine) -> list[Action] | None:
        self.num_queries += 1
        for stage in self.stages:
            cex = stage.find_counterexample(hypothesis)
            if self.verbose:
                name = type(stage).__name__
                print(f"  [{name}] {'CEX ' + str(cex) if cex else 'no CEX'}")
            if cex is not None:
                return cex
        return None
