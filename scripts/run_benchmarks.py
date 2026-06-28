"""Reproducible benchmark suite for the paper's results tables.

Runs the harness on each game-theoretic environment (PALS + ablations vs the
baselines, scored against random / greedy / optimal opponents), a noise-
sensitivity sweep over the preference oracle, and two active-shielding
experiments (the gas grid and a FrozenLake corridor), printing fixed-width
tables.

    python -m scripts.run_benchmarks          # all games + shielding
    python -m scripts.run_benchmarks --quick  # smaller budgets

Deterministic given the fixed seed.
"""

from __future__ import annotations

import argparse
import random
import time

from pals.bench.evaluate import evaluate_vs_opponents, play_game
from pals.bench.harness import benchmark, format_table, standard_opponents
from pals.bench.noisy import NoisyOracle
from pals.bench.players import PALSPlayer, RandomPlayer, greedy_action
from pals.core.learner import run_pals
from pals.core.preference import MinimaxPreferenceOracle
from pals.envs.base import Player
from pals.envs.dots_and_boxes import DotsAndBoxesEnv, score_margin_heuristic
from pals.envs.frozen_lake import (
    FrozenLakeEnv,
    in_hole_predicate,
    manhattan_progress_heuristic,
    safe_goal_action,
)
from pals.envs.frozen_lake import safety_state_key as hole_safety_key
from pals.envs.gas_grid import (
    GasGridEnv,
    gas_depleted,
    manhattan_greedy_heuristic,
    safety_state_key,
)
from pals.envs.minimax import MinimaxEnv, leftmost_leaf_heuristic
from pals.envs.nim import NimEnv, largest_pile_heuristic
from pals.envs.tic_tac_toe import TicTacToeEnv, line_control_heuristic
from pals.shielding.model_check import find_violation
from pals.shielding.spec import SafetySpec

SEED = 0


def _game_rows(name, env, heuristic, *, oracle_depth, quick):
    print(f"\n### {name}")
    oracle = MinimaxPreferenceOracle(env, heuristic, depth=oracle_depth)
    rows = benchmark(
        env,
        heuristic,
        oracle,
        n_games=10 if quick else 30,
        depth_n=2,
        rollout_budget=30 if quick else 80,
        uct_budget=50 if quick else 150,
        qlearn_episodes=1000 if quick else 4000,
        rng=random.Random(SEED),
    )
    print(format_table(rows))


def run_games(quick: bool) -> None:
    _game_rows(
        "Nim (1,2,3)",
        NimEnv(piles=(1, 2, 3)),
        largest_pile_heuristic,
        oracle_depth=1,
        quick=quick,
    )
    _game_rows(
        "Minimax (depth 6, b=2)",
        MinimaxEnv(depth=6, branching=2, seed=SEED),
        leftmost_leaf_heuristic,
        oracle_depth=1,
        quick=quick,
    )
    _game_rows(
        "Tic-Tac-Toe",
        TicTacToeEnv(),
        line_control_heuristic,
        oracle_depth=2,
        quick=quick,
    )
    _game_rows(
        "Dots & Boxes (2x2)",
        DotsAndBoxesEnv(rows=2, cols=2),
        score_margin_heuristic,
        oracle_depth=2,
        quick=quick,
    )


def _noise_rows(name, env, heuristic, *, oracle_depth, quick):
    """Sweep oracle noise and report, per config, the learned automaton size,
    accepted audit deviations, and play quality vs each opponent.

    Contrasts the full PALS (audit on) against the L*+PAC core (audit off) under
    a *consistently imperfect* teacher (``NoisyOracle`` corrupts a fixed fraction
    of answers, deterministically per input). The point R1 asked about: the
    L*+PAC core stays small and robust as noise grows, whereas the MCTS audit
    accepts ever more deviations (its Thm-2 termination assumes consistent
    preferences, which noise violates) and play quality degrades.
    """
    print(f"\n### Noise sensitivity — {name}")
    print(
        f"{'config':<14}{'noise':>6}{'states':>8}{'devs':>6}"
        f"{'vs_random':>11}{'vs_greedy':>11}{'vs_optimal':>12}"
    )
    print("-" * 68)
    noises = (0.0, 0.25) if quick else (0.0, 0.1, 0.25, 0.5)
    opponents = standard_opponents(heuristic)
    n_games = 15 if quick else 30
    for use_mcts in (True, False):
        config = "PALS" if use_mcts else "PALS_no_mcts"
        for noise in noises:
            inner = MinimaxPreferenceOracle(env, heuristic, depth=oracle_depth)
            oracle = NoisyOracle(inner, env, noise=noise, seed=SEED)
            result = run_pals(
                env,
                oracle,
                use_mcts=use_mcts,
                use_pac=True,
                depth_n=2,
                rollout_budget=30 if quick else 80,
                rng=random.Random(SEED),
            )
            scores = evaluate_vs_opponents(
                env,
                PALSPlayer(result.model),
                opponents,
                n_games=n_games,
                rng=random.Random(SEED),
            )
            print(
                f"{config:<14}{noise:>6.2f}{len(result.model.states):>8}"
                f"{result.accepted_deviations:>6}"
                f"{scores['vs_random']:>11.3f}{scores['vs_greedy']:>11.3f}"
                f"{scores['vs_optimal']:>12.3f}"
            )


def run_noise(quick: bool) -> None:
    # Fast games only: under high noise the audit's deviation count (and so the
    # automaton, and runtime) blows up on large-alphabet games like Tic-Tac-Toe
    # — that blowup is documented in docs/03_paper_alignment.md §4 rather than
    # baked into the default suite.
    _noise_rows(
        "Nim (1,2,3)",
        NimEnv(piles=(1, 2, 3)),
        largest_pile_heuristic,
        oracle_depth=1,
        quick=quick,
    )
    _noise_rows(
        "Minimax (depth 6, b=2)",
        MinimaxEnv(depth=6, branching=2, seed=SEED),
        leftmost_leaf_heuristic,
        oracle_depth=1,
        quick=quick,
    )


# Gas-grid shielding parameterizations: (rows, cols, home, refuel, dropoff,
# gas_max). gas_is_fatal=False keeps the preference oracle gas-blind, so safety
# (G(gas>0)) is misaligned with preference and only the shield can enforce it.
# Constraints (enforced by _validate_gas_config): home/refuel/dropoff are
# distinct and on-grid, and gas_max is large enough that the task is winnable —
# it scales with the grid rather than staying at the 1x3 corridor's value of 2.
GasGridConfig = dict
GAS_GRID_CONFIGS: tuple[GasGridConfig, ...] = (
    {
        "rows": 1,
        "cols": 3,
        "home": (0, 0),
        "refuel": (0, 1),
        "dropoff": (0, 2),
        "gas_max": 3,
    },  # noqa: E501
    {
        "rows": 2,
        "cols": 2,
        "home": (0, 0),
        "refuel": (1, 0),
        "dropoff": (1, 1),
        "gas_max": 4,
    },  # noqa: E501
    {
        "rows": 3,
        "cols": 2,
        "home": (0, 1),
        "refuel": (1, 0),
        "dropoff": (2, 1),
        "gas_max": 6,
    },  # noqa: E501
    {
        "rows": 3,
        "cols": 3,
        "home": (2, 0),
        "refuel": (1, 1),
        "dropoff": (0, 2),
        "gas_max": 9,
    },  # noqa: E501
    {
        "rows": 3,
        "cols": 3,
        "home": (0, 0),
        "refuel": (2, 1),
        "dropoff": (2, 2),
        "gas_max": 9,
    },  # noqa: E501
    {
        "rows": 4,
        "cols": 4,
        "home": (0, 0),
        "refuel": (2, 2),
        "dropoff": (3, 3),
        "gas_max": 16,
        "use_pac": True,
        "rollout_budget": 50,
        "depth_n": 4,
    },  # noqa: E501
)


def _manhattan(p: tuple[int, int], q: tuple[int, int]) -> int:
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def _validate_gas_config(cfg: GasGridConfig) -> None:
    """Reject overlapping/off-grid locations and a tank too small to ever win."""
    rows, cols = cfg["rows"], cfg["cols"]
    locs = {"home": cfg["home"], "refuel": cfg["refuel"], "dropoff": cfg["dropoff"]}
    for name, (r, c) in locs.items():
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"{name} {(r, c)} off the {rows}x{cols} grid")
    if len(set(locs.values())) != 3:
        raise ValueError(f"home/refuel/dropoff must be distinct: {locs}")
    # Winnable iff some pickup->deliver path fits the tank: either a direct
    # home->dropoff leg, or two legs each refilled at the refuel cell.
    direct = _manhattan(cfg["home"], cfg["dropoff"])
    via_refuel = max(
        _manhattan(cfg["home"], cfg["refuel"]),
        _manhattan(cfg["refuel"], cfg["dropoff"]),
    )
    need = min(direct, via_refuel)
    if cfg["gas_max"] < need:
        raise ValueError(
            f"gas_max={cfg['gas_max']} cannot win on {rows}x{cols} (needs >= {need})"
        )


def _run_gas_grid(cfg: GasGridConfig) -> dict:
    """Run the unshielded/shielded comparison for one parameterization; return a
    result row (config, safety outcomes, delivery, success, runtime)."""
    _validate_gas_config(cfg)
    env = GasGridEnv(
        rows=cfg["rows"],
        cols=cfg["cols"],
        home=cfg["home"],
        refuel=cfg["refuel"],
        dropoff=cfg["dropoff"],
        gas_max=cfg["gas_max"],
        max_steps=max(20, 4 * (cfg["rows"] + cfg["cols"])),
        eligible_cells=(cfg["home"],),
        gas_is_fatal=False,
    )
    oracle = MinimaxPreferenceOracle(env, manhattan_greedy_heuristic, depth=4)
    spec = SafetySpec(gas_depleted, name="G(gas>0)", state_key=safety_state_key)

    # Learning budget defaults suit the small grids; larger grids opt into more.
    # The 4x4 needs deeper audit lookahead (depth_n): at depth_n=2 the audit only
    # rolls out one step and can't discover the multi-step navigate-then-DROP path
    # to the deep drop-off. A larger rollout_budget and PAC alone don't help (the
    # audit terminates early finding no improving deviation) — depth_n is the lever.
    use_pac = cfg.get("use_pac", False)
    rollout_budget = cfg.get("rollout_budget", 10)
    depth_n = cfg.get("depth_n", 2)

    start = time.perf_counter()
    unshielded = run_pals(
        env,
        oracle,
        depth_n=depth_n,
        rollout_budget=rollout_budget,
        use_pac=use_pac,
        rng=random.Random(SEED),
    )
    shielded = run_pals(
        env,
        oracle,
        depth_n=depth_n,
        rollout_budget=rollout_budget,
        use_pac=use_pac,
        spec=spec,
        prefer_action=greedy_action(env, manhattan_greedy_heuristic),
        rng=random.Random(SEED),
    )
    runtime = time.perf_counter() - start

    unsafe = find_violation(env, unshielded.model, gas_depleted) is not None
    safe = find_violation(env, shielded.model, gas_depleted) is not None
    delivers = play_game(
        env, RandomPlayer(), PALSPlayer(shielded.model), random.Random(SEED)
    )
    return {
        "label": (
            f"{env.rows}x{env.cols} home={cfg['home']} refuel={cfg['refuel']} "
            f"dropoff={cfg['dropoff']} gas={env.gas_max}"
        ),
        "unshielded_violation": unsafe,
        "shielded_violation": safe,
        "patches": shielded.shield_patches,
        "delivers": delivers,
        "success": (not safe) and delivers == 1.0,
        "runtime_s": runtime,
    }


def run_gas_grid_shielding() -> list[dict]:
    """Run every gas-grid parameterization; print a table and return the rows."""
    print("\n### Active shielding — gas grid, spec G(gas>0)")
    header = (
        f"{'config':<54}{'unshield':>9}{'shield':>8}{'patch':>6}"
        f"{'deliver':>8}{'ok':>4}{'time_s':>8}"
    )
    print(header)
    print("-" * len(header))
    results = []
    for cfg in GAS_GRID_CONFIGS:
        r = _run_gas_grid(cfg)
        results.append(r)
        print(
            f"{r['label']:<54}{str(r['unshielded_violation']):>9}"
            f"{str(r['shielded_violation']):>8}{r['patches']:>6}"
            f"{r['delivers']:>8.1f}{str(r['success']):>4}{r['runtime_s']:>8.1f}"
        )
    return results


def _frozenlake_route(env, model, spawn):
    """Drive the learned controller deterministically from ``spawn`` to a terminal
    state; return ``(p2_moves, terminal_reward)``."""
    state = env.step(env.initial_state(), ("SPAWN", spawn))
    trace = [("SPAWN", spawn)]
    moves = []
    player = PALSPlayer(model)
    while not env.is_terminal(state):
        if env.current_player(state) is Player.P1:
            obs = env.legal_actions(state)[0]
            state = env.step(state, obs)
            trace.append(obs)
        else:
            action = player.action(env, trace, random.Random(SEED))
            state = env.step(state, action)
            trace.append(action)
            moves.append(action)
    return moves, env.reward(state)


# FrozenLake ("ice game") shielding parameterizations — the same multi-config
# treatment as the gas grid. Each is a map where the hole-blind greedy path
# crosses a hole (S start, H hole, G goal, F frozen); holes_are_fatal=False makes
# the oracle hole-blind so only the shield (G(not hole)) can enforce safety. No
# 4x4 here, matching the gas-grid suite, to keep the run quick.
FROZEN_LAKE_CONFIGS: tuple[tuple[str, ...], ...] = (
    ("SHG", "FFF"),  # 2x3 corridor
    ("SHG", "FFF", "FFF"),  # 3x3
    ("SHFG", "FFFF"),  # 2x4 corridor
)


def _spawn_cell(env: FrozenLakeEnv, desc: tuple[str, ...]) -> tuple[int, int]:
    """The 'S' tile if present (else the first solvable start cell)."""
    for r, row in enumerate(desc):
        if "S" in row:
            return (r, row.index("S"))
    return env.start_cells[0]


def _run_frozenlake(desc: tuple[str, ...]) -> dict:
    """Run the unshielded/shielded comparison for one FrozenLake map; return a
    result row (config, safety outcomes, reward, success, runtime)."""
    env = FrozenLakeEnv(desc=desc, holes_are_fatal=False)
    heuristic = manhattan_progress_heuristic
    is_hole = in_hole_predicate(env)
    spec = SafetySpec(is_hole, name="G(not hole)", state_key=hole_safety_key)
    spawn = _spawn_cell(env, desc)

    start = time.perf_counter()
    unshielded = run_pals(
        env,
        MinimaxPreferenceOracle(env, heuristic, depth=4),
        depth_n=2,
        rollout_budget=10,
        use_pac=False,
        rng=random.Random(SEED),
    )
    shielded = run_pals(
        env,
        MinimaxPreferenceOracle(env, heuristic, depth=4),
        depth_n=2,
        rollout_budget=10,
        use_pac=False,
        spec=spec,
        prefer_action=safe_goal_action(env),
        rng=random.Random(SEED),
    )
    runtime = time.perf_counter() - start

    unsafe = find_violation(env, unshielded.model, is_hole) is not None
    safe = find_violation(env, shielded.model, is_hole) is not None
    _, s_reward = _frozenlake_route(env, shielded.model, spawn)
    return {
        "label": f"{env.rows}x{env.cols} {'/'.join(desc)} spawn={spawn}",
        "unshielded_violation": unsafe,
        "shielded_violation": safe,
        "patches": shielded.shield_patches,
        "reward": s_reward,
        "success": (not safe) and s_reward == 1.0,
        "runtime_s": runtime,
    }


def run_frozenlake_shielding() -> list[dict]:
    """Run every FrozenLake parameterization; print a table and return the rows."""
    print("\n### Active shielding #2 — FrozenLake (ice game), spec G(not hole)")
    header = (
        f"{'config':<34}{'unshield':>9}{'shield':>8}{'patch':>6}"
        f"{'reward':>8}{'ok':>4}{'time_s':>8}"
    )
    print(header)
    print("-" * len(header))
    results = []
    for desc in FROZEN_LAKE_CONFIGS:
        r = _run_frozenlake(desc)
        results.append(r)
        print(
            f"{r['label']:<34}{str(r['unshielded_violation']):>9}"
            f"{str(r['shielded_violation']):>8}{r['patches']:>6}"
            f"{r['reward']:>8.1f}{str(r['success']):>4}{r['runtime_s']:>8.1f}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_games(args.quick)
    run_noise(args.quick)
    run_gas_grid_shielding()
    run_frozenlake_shielding()


if __name__ == "__main__":
    main()
