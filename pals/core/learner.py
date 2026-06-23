"""Top-level PALS entry point: wire the pieces and run the learning loop.

Assembles the SUL, Table B, the equivalence-oracle chain (MCTS audit → PAC), and
Mealy L*, then runs it. Ablations are plain flags — drop the audit or PAC by
toggling ``use_mcts`` / ``use_pac`` — because the composite oracle treats every
stage uniformly.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pals.core.lstar import EquivalenceOracle, MealyLStar, MealyMachine
from pals.core.preference import PreferenceOracle
from pals.core.sul import PreferenceSUL
from pals.core.table_b import TableB
from pals.envs.base import Action, Environment, State
from pals.oracles.composite import CompositeEquivalenceOracle
from pals.oracles.mcts_audit import MCTSAuditOracle
from pals.oracles.pac import PACEquivalenceOracle
from pals.shielding.shield_oracle import ShieldOracle
from pals.shielding.spec import SafetySpec


@dataclass
class PALSResult:
    model: MealyMachine
    sul: PreferenceSUL
    table_b: TableB
    eq_oracle: EquivalenceOracle
    membership_queries: int
    accepted_deviations: int
    shield_patches: int


def run_pals(
    env: Environment,
    oracle: PreferenceOracle,
    depth_n: int = 4,
    rollout_budget: int = 200,
    use_mcts: bool = True,
    use_pac: bool = True,
    pac_eps: float = 0.05,
    pac_delta: float = 0.05,
    spec: SafetySpec | None = None,
    prefer_action: Callable[[State], Action | None] | None = None,
    rng: random.Random | None = None,
    verbose: bool = False,
    alphabet: Sequence[Action] | None = None,
) -> PALSResult:
    """Run PALS on ``env`` guided by ``oracle``; return the learned machine.

    Passing a ``spec`` adds the shielding stage that enforces it (the shield is
    placed last, after preference convergence).
    """
    rng = rng or random.Random()
    alphabet = list(alphabet) if alphabet is not None else list(env.p1_alphabet)

    sul = PreferenceSUL(env, oracle)
    table_b = TableB()

    stages: list[EquivalenceOracle] = []
    mcts: MCTSAuditOracle | None = None
    if use_mcts:
        mcts = MCTSAuditOracle(
            env,
            sul,
            oracle,
            table_b,
            depth_n=depth_n,
            rollout_budget=rollout_budget,
            rng=rng,
            verbose=verbose,
        )
        stages.append(mcts)
    if use_pac:
        stages.append(
            PACEquivalenceOracle(env, sul, eps=pac_eps, delta=pac_delta, rng=rng)
        )
    shield: ShieldOracle | None = None
    if spec is not None:
        shield = ShieldOracle(
            env, sul, spec, prefer_action=prefer_action, verbose=verbose
        )
        stages.append(shield)  # last: enforce safety after preference convergence
    if not stages:
        raise ValueError("run_pals needs at least one of use_mcts / use_pac / spec")

    eq_oracle: EquivalenceOracle = (
        stages[0]
        if len(stages) == 1
        else CompositeEquivalenceOracle(*stages, verbose=verbose)
    )

    model = MealyLStar(alphabet, sul, eq_oracle, verbose=verbose).run()
    return PALSResult(
        model=model,
        sul=sul,
        table_b=table_b,
        eq_oracle=eq_oracle,
        membership_queries=sul.num_queries,
        accepted_deviations=mcts.accepted_deviations if mcts else 0,
        shield_patches=shield.patches_installed if shield else 0,
    )
