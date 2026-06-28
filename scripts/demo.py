"""Human-facing demos for PALS — see the learned policy work, no code-reading required.

Two subcommands, both driving the *same* code paths the tests and benchmarks
use (``run_pals`` to learn, ``PALSPlayer`` to act, ``find_violation`` to model
check) — so if a demo plays correctly, the audited pipeline is what produced it.

    # Play against the learned controller (games: nim, ttt, dots, minimax):
    python -m scripts.demo play nim
    python -m scripts.demo play ttt --show-policy     # also print the learned DFA

    # Watch the controller play itself, no typing (handy for a quick sanity check):
    python -m scripts.demo play minimax --opponent optimal

    # Walk the agent through the grid (press Enter to advance each step) to watch
    # the shielding layer turn an unsafe policy into a safe one:
    python -m scripts.demo shield gas
    python -m scripts.demo shield frozenlake
    python -m scripts.demo shield gas --only unshielded   # just the no-shield run
    python -m scripts.demo shield gas --delay 0.7         # auto-advance, not Enter

Deterministic given the fixed seed.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from pals.bench.players import (
    PALSPlayer,
    RandomPlayer,
    greedy_action,
    optimal_player,
)
from pals.core.learner import run_pals
from pals.core.lstar import MealyMachine
from pals.core.preference import MinimaxPreferenceOracle
from pals.envs.base import Environment, Player
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
from pals.envs.nim import NimEnv, NimState, largest_pile_heuristic
from pals.envs.tic_tac_toe import TicTacToeEnv, TTTState, line_control_heuristic
from pals.shielding.model_check import find_violation
from pals.shielding.spec import SafetySpec

SEED = 0


# ======================================================================
# Pretty-printing per environment (state render + action labels)
# ======================================================================


def _render_nim(state: NimState) -> str:
    rows = [
        f"  pile {i}: {'|' * n}{'  (empty)' if n == 0 else f'  ({n})'}"
        for i, n in enumerate(state.piles)
    ]
    return "\n".join(rows)


def _label_nim(action) -> str:
    pile, count = action
    return f"take {count} from pile {pile}"


def _render_ttt(state: TTTState) -> str:
    b = [c if c else str(i) for i, c in enumerate(state.board)]
    sep = "\n  ---+---+---\n"
    return "  " + sep.join(f" {b[r]} | {b[r + 1]} | {b[r + 2]} " for r in (0, 3, 6))


def _label_ttt(action) -> str:
    return f"play cell {action}"


def _make_render_dots(env: DotsAndBoxesEnv):
    """Render the edge grid; undrawn edges show their number (so you can pick
    them), drawn edges are lines, and completed boxes get a ``*``. Edge numbers
    are single-digit on the small default board, keeping the grid aligned."""

    def render(state) -> str:
        drawn = state.drawn
        lines = []
        for r in range(env.rows + 1):
            dots = "  o"
            for c in range(env.cols):
                e = r * env.cols + c
                dots += ("---" if e in drawn else f" {e} ") + "o"
            lines.append(dots)
            if r < env.rows:
                mids = "  "
                for c in range(env.cols + 1):
                    ve = env._n_horizontal + r * (env.cols + 1) + c
                    mids += "|" if ve in drawn else str(ve)
                    if c < env.cols:
                        box = r * env.cols + c
                        filled = all(x in drawn for x in env._box_edges[box])
                        mids += " * " if filled else "   "
                lines.append(mids)
        lines.append(f"  score  PALS/P2={state.p2_score}  you/P1={state.p1_score}")
        return "\n".join(lines)

    return render


def _label_dots(action) -> str:
    return f"draw edge {action}"


def _make_render_minimax(env: MinimaxEnv):
    """Show the path taken, whose turn (P1 minimizes, P2 maximizes), and the leaf
    payoffs still reachable from here — so you can see PALS steer toward high ones."""

    def render(state) -> str:
        head = f"  path={list(state)}  depth {len(state)}/{env.depth}"
        if env.is_terminal(state):
            return f"{head}  ->  leaf payoff {env.reward(state):.1f}"
        role = (
            "you/P1 want LOW"
            if env.current_player(state) is Player.P1
            else "PALS/P2 wants HIGH"
        )
        reach = sorted(
            round(v, 1) for p, v in env._leaf_values.items() if p[: len(state)] == state
        )
        return f"{head}  ({role})\n  reachable leaf payoffs: {reach}"

    return render


def _label_minimax(action) -> str:
    return f"take branch {action}"


# game name -> (env factory, heuristic, render *factory* (env -> state -> str),
#               action labeller, blurb)
GAMES = {
    "nim": (
        lambda: NimEnv(piles=(1, 2, 3)),
        largest_pile_heuristic,
        lambda env: _render_nim,
        _label_nim,
        "Normal-play Nim (piles 1,2,3): take the last object to win.",
    ),
    "ttt": (
        TicTacToeEnv,
        line_control_heuristic,
        lambda env: _render_ttt,
        _label_ttt,
        "Tic-Tac-Toe: you are X (P1), PALS is O (P2). Three in a row wins.",
    ),
    "dots": (
        lambda: DotsAndBoxesEnv(rows=1, cols=2),
        score_margin_heuristic,
        _make_render_dots,
        _label_dots,
        "Dots & Boxes (1x2): draw edges, complete a box to score. Most boxes wins.",
    ),
    "minimax": (
        lambda: MinimaxEnv(depth=4, branching=2, seed=SEED),
        leftmost_leaf_heuristic,
        _make_render_minimax,
        _label_minimax,
        "Minimax tree (depth 4): you minimize the leaf payoff, PALS maximizes it.",
    ),
}


# ======================================================================
# `play` — a human (or a baseline) vs the learned PALS controller
# ======================================================================


def _print_policy(model: MealyMachine, label) -> None:
    print("\n  Learned symbolic policy (Mealy machine):")
    print(f"    states: {len(model.states)}   initial: {model.initial_state.state_id}")
    for st in model.states:
        for inp, (out, nxt) in st.transitions.items():
            if out is None:
                continue  # transition into a terminal state — no P2 response
            print(
                f"    [{st.state_id}] --on P1 {inp}--> respond '{label(out)}' "
                f"-> [{nxt.state_id}]"
            )
    print()


def _human_action(env: Environment, trace, label) -> object:
    legal = env.legal_actions(env.get_node(trace))
    print("\n  Your move (P1). Options:")
    for i, a in enumerate(legal):
        print(f"    [{i}] {label(a)}")
    while True:
        raw = input("  pick a number: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(legal):
            return legal[int(raw)]
        print("  invalid — try again.")


def cmd_play(game: str, opponent: str | None, show_policy: bool) -> None:
    env_factory, heuristic, render_factory, label, blurb = GAMES[game]
    env = env_factory()
    render = render_factory(env)
    rng = random.Random(SEED)

    print(f"\n=== PALS demo: {blurb} ===")
    print("Learning a symbolic policy from a *suboptimal* preference oracle...")
    oracle = MinimaxPreferenceOracle(env, heuristic, depth=1)
    result = run_pals(env, oracle, depth_n=4, rollout_budget=200, rng=rng)
    pals = PALSPlayer(result.model)
    print(
        f"  done: {len(result.model.states)}-state policy, "
        f"{result.accepted_deviations} MCTS-audit improvements over the oracle."
    )
    if show_policy:
        _print_policy(result.model, label)

    # P1 = you (or a baseline opponent); P2 = PALS.
    auto = None
    if opponent == "random":
        auto = RandomPlayer()
    elif opponent == "optimal":
        auto = optimal_player(heuristic)

    trace: list = []
    state = env.initial_state()
    while not env.is_terminal(state):
        print("\n" + render(state))
        if env.current_player(state) is Player.P1:
            if auto is not None:
                action = auto.action(env, trace, rng)
                print(f"\n  P1 ({opponent}) plays: {label(action)}")
            else:
                action = _human_action(env, trace, label)
        else:
            action = pals.action(env, trace, rng)
            print(f"\n  PALS (P2) responds: {label(action)}")
        trace.append(action)
        state = env.step(state, action)

    print("\n" + render(state))
    reward = env.reward(state)
    verdict = {1.0: "PALS (P2) wins", -1.0: "P1 wins", 0.0: "draw"}.get(
        reward, f"reward {reward}"
    )
    print(f"\n=== Result: {verdict} (P2 reward = {reward}) ===\n")


# ======================================================================
# `shield` — *watch* an unsafe policy get patched into a safe one
#
# The agent walks the grid frame-by-frame: you see the unshielded controller
# run the tank to zero / step into the hole, then the shielded one detour or
# refuel. Rendering is a plain ASCII map so it animates in any terminal.
# ======================================================================

_CLEAR = "\033[2J\033[3J\033[H"  # clear screen + scrollback, home the cursor


def _trajectory(env: Environment, model: MealyMachine) -> list:
    """Drive ``model`` (P2) against the forced/first P1 inputs and return the
    sequence of settled agent states (the start cell, then one per P2 move)."""
    player = PALSPlayer(model)
    state = env.initial_state()
    trace: list = []
    frames: list = []
    while True:
        pos = getattr(state, "pos", None)
        settled = env.is_terminal(state) or env.current_player(state) is Player.P1
        if settled and pos is not None and (not frames or frames[-1] is not state):
            frames.append(state)
        if env.is_terminal(state):
            return frames
        if env.current_player(state) is Player.P1:
            action = env.legal_actions(state)[0]
        else:
            action = player.action(env, trace, random.Random(SEED))
        trace.append(action)
        state = env.step(state, action)


def _grid(env, state, cell_char) -> str:
    """Box-drawn grid where ``cell_char(r, c)`` gives each cell's glyph; the agent
    ('A') is overlaid by ``cell_char`` itself."""
    border = "  +" + "+".join("---" for _ in range(env.cols)) + "+"
    out = [border]
    for r in range(env.rows):
        cells = "|".join(f" {cell_char(r, c)} " for c in range(env.cols))
        out.append(f"  |{cells}|")
        out.append(border)
    return "\n".join(out)


def _render_gas(env: GasGridEnv, state) -> str:
    def cell(r, c):
        if state.pos == (r, c):
            return "A"
        return {env.home: "S", env.refuel: "R", env.dropoff: "D"}.get((r, c), ".")

    filled = round(state.gas / env.gas_max * 6)
    gauge = "[" + "#" * filled + " " * (6 - filled) + f"] {state.gas}/{env.gas_max}"
    status = f"  gas: {gauge}   carrying: {'yes' if state.carrying else 'no'}"
    if state.delivered:
        status += "   DELIVERED"
    if gas_depleted(state):
        status += "   <-- VIOLATION G(gas>0)"
    legend = "  S=start  R=refuel  D=dropoff  A=agent"
    return f"{_grid(env, state, cell)}\n{status}\n{legend}"


def _render_frozenlake(env: FrozenLakeEnv, state) -> str:
    def cell(r, c):
        return "A" if state.pos == (r, c) else env.desc[r][c]

    status = "  on goal!" if state.pos == env.goal else ""
    if state.pos in env.holes:
        status = "  <-- VIOLATION G(not hole)"
    legend = "  S=start  F=frozen  H=hole  G=goal  A=agent"
    return f"{_grid(env, state, cell)}\n{status}\n{legend}"


def _animate(title, env, frames, render, summary, delay) -> None:
    """Show ``frames`` one at a time. ``delay is None`` waits for Enter between
    frames (self-paced — the default); ``delay > 0`` auto-advances on a timer;
    ``delay == 0`` prints every frame at once (no clearing)."""
    n = len(frames)
    step = delay is None
    clear = step or bool(delay)
    for i, state in enumerate(frames, 1):
        if clear:
            print(_CLEAR, end="")
        print(f"\n  {title}  (step {i}/{n})")
        print(render(env, state))
        last = i == n
        if step and not last:
            try:
                input("\n  [Enter] next ▸ ")
            except EOFError:
                step = False  # not a TTY (piped): show the rest without pausing
        elif not step and delay and delay > 0:
            sys.stdout.flush()
            time.sleep(delay)
        elif not step and not last:
            print()
    print(summary + "\n")


def _shield_gas(delay: float, only: str) -> None:
    print("\n=== Shielding demo: gas corridor, spec G(gas > 0) ===")
    print("Corridor S(start)=refuel-home  R=refuel  D=dropoff, tank = 2.")
    print("The preference oracle is *gas-blind* — it just heads for the dropoff.")
    env = GasGridEnv(
        rows=1,
        cols=3,
        home=(0, 0),
        refuel=(0, 1),
        dropoff=(0, 2),
        gas_max=2,
        eligible_cells=((0, 0),),
        gas_is_fatal=False,
    )
    oracle = MinimaxPreferenceOracle(env, manhattan_greedy_heuristic, depth=4)
    spec = SafetySpec(gas_depleted, name="G(gas>0)", state_key=safety_state_key)
    common = {"depth_n": 2, "rollout_budget": 10, "use_pac": False}

    if only in ("both", "unshielded"):
        unshielded = run_pals(env, oracle, **common, rng=random.Random(SEED))
        u_bad = find_violation(env, unshielded.model, gas_depleted) is not None
        _animate(
            "UNSHIELDED",
            env,
            _trajectory(env, unshielded.model),
            _render_gas,
            f"  => reachable G(gas>0) violation: {u_bad}  (runs the tank dry)",
            delay,
        )
    if only in ("both", "shielded"):
        shielded = run_pals(
            env,
            oracle,
            **common,
            spec=spec,
            prefer_action=greedy_action(env, manhattan_greedy_heuristic),
            rng=random.Random(SEED),
        )
        s_bad = find_violation(env, shielded.model, gas_depleted) is not None
        _animate(
            "SHIELDED",
            env,
            _trajectory(env, shielded.model),
            _render_gas,
            f"  => reachable G(gas>0) violation: {s_bad}  "
            f"(shield added {shielded.shield_patches} refuel patch, still delivers)",
            delay,
        )


def _shield_frozenlake(delay: float, only: str) -> None:
    print("\n=== Shielding demo: FrozenLake 'SHG'/'FFF', spec G(not hole) ===")
    print("Top row S(start) H(hole) G(goal); the straight path crosses the hole.")
    print("The preference oracle is *hole-blind* — only the shield avoids it.")
    env = FrozenLakeEnv(desc=("SHG", "FFF"), holes_are_fatal=False)
    heuristic = manhattan_progress_heuristic
    is_hole = in_hole_predicate(env)
    spec = SafetySpec(is_hole, name="G(not hole)", state_key=hole_safety_key)
    common = {"depth_n": 2, "rollout_budget": 10, "use_pac": False}

    if only in ("both", "unshielded"):
        unshielded = run_pals(
            env,
            MinimaxPreferenceOracle(env, heuristic, depth=4),
            **common,
            rng=random.Random(SEED),
        )
        u_bad = find_violation(env, unshielded.model, is_hole) is not None
        _animate(
            "UNSHIELDED",
            env,
            _trajectory(env, unshielded.model),
            _render_frozenlake,
            f"  => reachable G(not hole) violation: {u_bad}  (walks through the hole)",
            delay,
        )
    if only in ("both", "shielded"):
        shielded = run_pals(
            env,
            MinimaxPreferenceOracle(env, heuristic, depth=4),
            **common,
            spec=spec,
            prefer_action=safe_goal_action(env),
            rng=random.Random(SEED),
        )
        s_bad = find_violation(env, shielded.model, is_hole) is not None
        _animate(
            "SHIELDED",
            env,
            _trajectory(env, shielded.model),
            _render_frozenlake,
            f"  => reachable G(not hole) violation: {s_bad}  "
            f"(shield added {shielded.shield_patches} detour patch, reaches goal)",
            delay,
        )


def cmd_shield(which: str, delay: float, only: str) -> None:
    if which == "gas":
        _shield_gas(delay, only)
    else:
        _shield_frozenlake(delay, only)


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_play = sub.add_parser("play", help="play a game vs the learned PALS controller")
    p_play.add_argument("game", choices=sorted(GAMES))
    p_play.add_argument(
        "--opponent",
        choices=("random", "optimal"),
        help="auto-play P1 with this baseline instead of asking you to type",
    )
    p_play.add_argument(
        "--show-policy", action="store_true", help="print the learned Mealy machine"
    )

    p_shield = sub.add_parser("shield", help="watch the shielding layer patch a policy")
    p_shield.add_argument("which", choices=("gas", "frozenlake"))
    p_shield.add_argument(
        "--delay",
        type=float,
        default=None,
        help="auto-advance N seconds per frame (default: wait for Enter; "
        "--delay 0 prints every frame at once)",
    )
    p_shield.add_argument(
        "--only",
        choices=("both", "unshielded", "shielded"),
        default="both",
        help="show only one run (e.g. --only unshielded for the no-shield policy)",
    )

    args = parser.parse_args()
    if args.command == "play":
        cmd_play(args.game, args.opponent, args.show_policy)
    else:
        cmd_shield(args.which, args.delay, args.only)


if __name__ == "__main__":
    main()
