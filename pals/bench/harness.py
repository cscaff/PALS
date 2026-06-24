"""Config-driven benchmark harness.

Runs PALS (and ablations of it) plus the baseline players on one environment,
scores every strategy against a common set of P1 opponents, and returns a table
of rows. This is what produces the precise win/tie/loss comparison the reviewers
asked for, and — because the equivalence oracle is a composite — **ablations are
just flags** (``no_mcts`` drops the audit, ``no_pac`` drops PAC, ``shielded`` adds
the safety stage).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from pals.bench.evaluate import evaluate_vs_opponents
from pals.bench.players import (
    GamePlayer,
    PALSPlayer,
    QLearningPlayer,
    RandomPlayer,
    UCTPlayer,
    greedy_player,
    optimal_player,
)
from pals.core.learner import run_pals
from pals.core.preference import Heuristic, PreferenceOracle
from pals.envs.base import Action, Environment, State


@dataclass
class BenchmarkRow:
    name: str
    scores: dict[str, float]
    info: dict = field(default_factory=dict)


def standard_opponents(heuristic: Heuristic) -> dict[str, GamePlayer]:
    """The P1 opponents PALS is scored against (paper's vs_* columns)."""
    return {
        "vs_random": RandomPlayer(),
        "vs_greedy": greedy_player(heuristic),
        "vs_optimal": optimal_player(heuristic),
    }


# Default PALS ablations, expressed purely as run_pals flag overrides.
DEFAULT_VARIANTS: dict[str, dict] = {
    "PALS": {},
    "PALS_no_mcts": {"use_mcts": False, "use_pac": True},
    "PALS_no_pac": {"use_mcts": True, "use_pac": False},
}


def benchmark(
    env: Environment,
    heuristic: Heuristic,
    oracle: PreferenceOracle,
    *,
    opponents: Mapping[str, GamePlayer] | None = None,
    pals_variants: Mapping[str, dict] | None = None,
    include_baselines: bool = True,
    n_games: int = 20,
    depth_n: int = 2,
    rollout_budget: int = 50,
    uct_budget: int = 100,
    qlearn_episodes: int = 2000,
    prefer_action: Callable[[State], Action | None] | None = None,
    rng: random.Random | None = None,
) -> list[BenchmarkRow]:
    rng = rng or random.Random(0)
    opponents = opponents or standard_opponents(heuristic)
    variants = pals_variants if pals_variants is not None else DEFAULT_VARIANTS
    rows: list[BenchmarkRow] = []

    for name, overrides in variants.items():
        result = run_pals(
            env,
            oracle,
            depth_n=depth_n,
            rollout_budget=rollout_budget,
            prefer_action=prefer_action,
            rng=random.Random(rng.random()),
            **overrides,
        )
        scores = evaluate_vs_opponents(
            env, PALSPlayer(result.model), opponents, n_games=n_games, rng=rng
        )
        rows.append(
            BenchmarkRow(
                name=name,
                scores=scores,
                info={
                    "states": len(result.model.states),
                    "membership_queries": result.membership_queries,
                    "accepted_deviations": result.accepted_deviations,
                    "shield_patches": result.shield_patches,
                },
            )
        )

    if include_baselines:
        baselines: dict[str, GamePlayer] = {
            "Random": RandomPlayer(),
            "Greedy": greedy_player(heuristic),
            "Optimal": optimal_player(heuristic),
            "UCT": UCTPlayer(budget=uct_budget),
            "QLearning": QLearningPlayer(
                env, episodes=qlearn_episodes, rng=random.Random(rng.random())
            ),
        }
        for name, player in baselines.items():
            scores = evaluate_vs_opponents(
                env, player, opponents, n_games=n_games, rng=rng
            )
            rows.append(BenchmarkRow(name=name, scores=scores))

    return rows


def format_table(rows: list[BenchmarkRow]) -> str:
    """Render benchmark rows as a fixed-width text table."""
    if not rows:
        return "(no rows)"
    cols = list(rows[0].scores)
    header = f"{'strategy':<16}" + "".join(f"{c:>12}" for c in cols)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = "".join(f"{row.scores.get(c, float('nan')):>12.3f}" for c in cols)
        lines.append(f"{row.name:<16}{cells}")
    return "\n".join(lines)
