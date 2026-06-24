"""FrozenLake (Gymnasium toy-text) as a PALS :class:`Environment`.

A grid of tiles — ``S`` start, ``F`` frozen (safe), ``H`` hole, ``G`` goal — that
the agent must cross from start to goal without falling in a hole. We model the
**non-slippery** variant (``is_slippery=False``): moves are deterministic, which
keeps the learned policy a clean finite-state controller and aligns with PALS's
stationarity assumption. This is the toy-text companion to the gas-grid /
Taxi motivation: a goal-directed navigation task with avoid-the-hazard structure.

**Two-player (reactive) framing**, mirroring :mod:`pals.envs.gas_grid`:

* **P1 — environment.** At the start (idle) P1 chooses *which start cell the
  agent spawns on* among the solvable cells — the genuine environment choice that
  makes the controller reactive rather than a single memorized path, and gives
  the ``vs_random/greedy/optimal`` opponents something to vary. On every later P1
  turn its only legal input is the **observation** (the agent's cell) — a forced
  singleton that reveals the position to the controller.
* **P2 — system.** The controller PALS learns; emits a move
  (``LEFT/DOWN/RIGHT/UP``, the Gymnasium action order).

Off-grid moves keep the agent in place (Gymnasium behaviour). The episode ends on
the goal (reward ``+1``), a hole (``-1``), or a step-count timeout (``-1``); the
``-1`` on holes is what makes the preference oracle prefer to avoid them.

Setting ``holes_are_fatal=False`` makes holes passable and unpenalized — the
preference oracle becomes hole-blind, so the safety spec ``G(not hole)`` is
*misaligned* with preference. That is the regime for the second shielding
demonstration: only the shield (not preference optimization) can enforce safety.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

from pals.envs.base import Action, Environment, Player

# P2 (controller) action tokens, in Gymnasium's 0..3 order.
LEFT, DOWN, RIGHT, UP = "LEFT", "DOWN", "RIGHT", "UP"
_MOVES = {LEFT: (0, -1), DOWN: (1, 0), RIGHT: (0, 1), UP: (-1, 0)}

# The standard Gymnasium 4x4 map.
DEFAULT_MAP = (
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
)

Cell = tuple[int, int]


class FrozenLakeState(NamedTuple):
    pos: Cell | None  # None while idle (P1 has not chosen a start yet)
    player: Player
    step_count: int


class FrozenLakeEnv(Environment):
    def __init__(
        self,
        desc: tuple[str, ...] = DEFAULT_MAP,
        max_steps: int = 50,
        holes_are_fatal: bool = True,
    ) -> None:
        if not desc or any(len(row) != len(desc[0]) for row in desc):
            raise ValueError("desc must be a non-empty rectangular grid")
        self.desc = tuple(desc)
        self.rows = len(desc)
        self.cols = len(desc[0])
        self.max_steps = max_steps
        # When False, stepping on a hole neither ends the episode nor affects
        # reward — the preference oracle becomes hole-blind, so safety
        # (G(not hole)) is *misaligned* with preference and only the shield can
        # enforce it. Mirrors GasGridEnv.gas_is_fatal.
        self.holes_are_fatal = holes_are_fatal

        cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        self.holes = frozenset(c for c in cells if self._tile(c) == "H")
        goals = [c for c in cells if self._tile(c) == "G"]
        if len(goals) != 1:
            raise ValueError("desc must contain exactly one goal 'G'")
        self.goal = goals[0]

        # Start cells: non-hole, non-goal tiles from which the goal is reachable
        # (so the opponent cannot hand the controller an unsolvable spawn).
        self.start_cells = tuple(
            c
            for c in cells
            if c not in self.holes and c != self.goal and self._reaches_goal(c)
        )
        if not self.start_cells:
            raise ValueError("no start cell can reach the goal")

    def _tile(self, cell: Cell) -> str:
        return self.desc[cell[0]][cell[1]]

    def _in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def _reaches_goal(self, start: Cell) -> bool:
        """BFS over non-hole tiles to check the goal is reachable from ``start``."""
        seen = {start}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            if cell == self.goal:
                return True
            for dr, dc in _MOVES.values():
                nxt = (cell[0] + dr, cell[1] + dc)
                if (
                    self._in_bounds(nxt)
                    and nxt not in self.holes
                    and nxt not in seen
                ):
                    seen.add(nxt)
                    queue.append(nxt)
        return False

    # ------------------------------------------------------------------
    # Observation digest (the P1 input on active turns)
    # ------------------------------------------------------------------

    def observation(self, state: FrozenLakeState) -> Cell:
        assert state.pos is not None
        return state.pos

    # ------------------------------------------------------------------
    # Environment primitives
    # ------------------------------------------------------------------

    @property
    def p1_alphabet(self) -> list[Action]:
        spawns: list[Action] = [("SPAWN", c) for c in self.start_cells]
        observations: list[Action] = [
            (r, c) for r in range(self.rows) for c in range(self.cols)
        ]
        return [*spawns, *observations]

    def initial_state(self) -> FrozenLakeState:
        return FrozenLakeState(pos=None, player=Player.P1, step_count=0)

    def current_player(self, state: FrozenLakeState) -> Player:
        return state.player

    def legal_actions(self, state: FrozenLakeState) -> list[Action]:
        if self.is_terminal(state):
            return []
        if state.player is Player.P1:
            if state.pos is None:  # idle: P1 chooses the spawn cell
                return [("SPAWN", c) for c in self.start_cells]
            return [self.observation(state)]  # active: forced observation reveal
        # P2: only in-bounds moves are legal. Excluding off-grid "stay" no-ops
        # avoids degenerate self-loops the greedy oracle can get stuck on.
        return [
            a
            for a, (dr, dc) in _MOVES.items()
            if self._in_bounds(self._shift(state.pos, dr, dc))
        ]

    @staticmethod
    def _shift(pos: Cell | None, dr: int, dc: int) -> Cell:
        assert pos is not None
        return (pos[0] + dr, pos[1] + dc)

    def step(self, state: FrozenLakeState, action: Action) -> FrozenLakeState:
        if state.player is Player.P1:
            if isinstance(action, tuple) and action and action[0] == "SPAWN":
                return state._replace(pos=action[1], player=Player.P2)
            return state._replace(player=Player.P2)  # observation reveal

        dr, dc = _MOVES[action]
        return FrozenLakeState(
            pos=self._shift(state.pos, dr, dc),
            player=Player.P1,
            step_count=state.step_count + 1,
        )

    def is_terminal(self, state: FrozenLakeState) -> bool:
        if state.pos is None:
            return False
        if state.pos == self.goal or state.step_count >= self.max_steps:
            return True
        return self.holes_are_fatal and state.pos in self.holes

    def reward(self, state: FrozenLakeState) -> float:
        return 1.0 if state.pos == self.goal else -1.0


def manhattan_progress_heuristic(env: FrozenLakeEnv, state: FrozenLakeState) -> float:
    """A deliberately *suboptimal* preference heuristic: rewards Manhattan
    progress toward the goal and is **hole-blind** — it will happily step toward a
    hole if it lies on the straight-line path, exactly the locally-greedy,
    not-globally-safe teacher PALS is meant to improve on (any hole-aversion comes
    from the environment reward when holes are fatal, not from this heuristic).
    Scaled into ``[-1, 1]`` from P2's perspective."""
    if state.pos is None:
        return 0.0
    if state.pos == env.goal:
        return 1.0
    span = env.rows + env.cols
    d = abs(state.pos[0] - env.goal[0]) + abs(state.pos[1] - env.goal[1])
    return 1.0 - 2.0 * d / max(1, span)


def in_hole_predicate(env: FrozenLakeEnv):
    """Build a ``state -> bool`` hole predicate bound to ``env`` — the safety spec
    ``G(not hole)`` for the shielding layer."""

    def predicate(state: FrozenLakeState) -> bool:
        return state.pos is not None and state.pos in env.holes

    return predicate


def safety_state_key(state: FrozenLakeState) -> tuple:
    """Abstraction for the hole spec: drop ``step_count`` (the spec ignores it),
    keeping the safety state space finite and patches state-keyed."""
    return (state.pos, state.player)


def safe_goal_action(env: FrozenLakeEnv):
    """A ``state -> action`` policy that steps along a **hole-free** shortest path
    to the goal. Used as the shield's ``prefer_action`` tie-breaker so the
    shielded controller is safe *and* still reaches the goal — the hole-blind
    greedy heuristic alone produces a safe-but-useless controller, because the
    detour around a hole briefly increases Manhattan distance to the goal."""
    dist: dict[Cell, int] = {env.goal: 0}
    queue = deque([env.goal])
    while queue:
        cell = queue.popleft()
        for dr, dc in _MOVES.values():
            nb = (cell[0] - dr, cell[1] - dc)
            if env._in_bounds(nb) and nb not in env.holes and nb not in dist:
                dist[nb] = dist[cell] + 1
                queue.append(nb)

    def choose(state: FrozenLakeState) -> Action | None:
        if (
            state.pos is None
            or env.is_terminal(state)
            or env.current_player(state) is not Player.P2
        ):
            return None
        best: Action | None = None
        best_d: int | None = None
        for action, (dr, dc) in _MOVES.items():
            nxt = (state.pos[0] + dr, state.pos[1] + dc)
            d = dist.get(nxt)
            if d is None:  # off-grid, a hole, or cannot reach the goal safely
                continue
            if best_d is None or d < best_d:
                best_d, best = d, action
        return best

    return choose
