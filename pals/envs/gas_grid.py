"""Gas-grid delivery robot — the paper's Figure 1 motivating example.

A robot on an ``rows x cols`` grid must pick up a package and deliver it to a
drop-off cell **without running out of gas**; it may refuel at a refuel cell.
This is the environment that exhibits the reward-vs-safety dichotomy the paper
motivates: a distance-greedy policy delivers fast but can run dry, while a safe
policy refuels when needed. The gas spec ``G(gas > 0)`` is the target for the
shielding layer (a later phase); the state carries ``gas`` precisely so that spec
can be checked.

**Two-player framing (reactive).** Following the modelling that makes the learned
policy a reactive controller rather than an open-loop plan:

* **P1 (environment).** At the start (idle) P1 chooses *which task arrives* — its
  cell is the genuine environment choice. On every later P1 turn its only legal
  input is the **observation digest** ``(pos, gas_band, task_loc, carrying)`` —
  a forced singleton that *reveals the current observation to the controller*.
* **P2 (system).** The controller; emits a move / pickup / drop / refuel. This is
  what PALS learns.

Because the per-step P1 input is the observation, the learned Mealy machine reads
the observation and reacts — a closed-loop controller. The full set of task
events plus observation digests is the L* input alphabet. Gas is quantized into
bands in the observation to bound that alphabet. Single-task episode: it ends on
delivery (success), gas depletion (failure), or a step-count timeout.
"""

from __future__ import annotations

from typing import NamedTuple

from pals.envs.base import Action, Environment, Player

# P2 (controller) action tokens.
N, S, E, W = "N", "S", "E", "W"
PICKUP, DROP, REFUEL = "PICKUP", "DROP", "REFUEL"

_MOVES = {N: (-1, 0), S: (1, 0), W: (0, -1), E: (0, 1)}

GAS_BANDS = ("critical", "low", "mid", "full")


def gas_band(gas: int, gas_max: int) -> str:
    """Coarse 4-band quantization of the gas level (bounds the alphabet)."""
    if gas <= 0:
        return "critical"
    if gas >= gas_max:
        return "full"
    frac = gas / gas_max
    if frac >= 0.5:
        return "mid"
    if frac >= 0.25:
        return "low"
    return "critical"


def manhattan(p: tuple[int, int], q: tuple[int, int]) -> int:
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


Cell = tuple[int, int]
Observation = tuple[Cell, str, "Cell | None", bool]
TaskEvent = tuple[str, Cell]


class GasGridState(NamedTuple):
    pos: Cell
    gas: int
    task_loc: Cell | None
    carrying: bool
    delivered: bool
    player: Player
    step_count: int


class GasGridEnv(Environment):
    def __init__(
        self,
        rows: int = 2,
        cols: int = 2,
        home: Cell = (0, 0),
        refuel: Cell | None = None,
        dropoff: Cell | None = None,
        gas_max: int = 4,
        max_steps: int = 20,
        eligible_cells: tuple[Cell, ...] | None = None,
        gas_is_fatal: bool = True,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.home = home
        self.refuel = refuel if refuel is not None else (0, cols - 1)
        self.dropoff = dropoff if dropoff is not None else (rows - 1, cols - 1)
        self.gas_max = gas_max
        self.max_steps = max_steps
        # When False, running out of gas neither ends the episode nor affects
        # reward — the preference oracle becomes gas-blind, so safety (G(gas>0))
        # is *misaligned* with preference and only the shield can enforce it.
        self.gas_is_fatal = gas_is_fatal
        if eligible_cells is None:
            eligible_cells = tuple(
                (r, c)
                for r in range(rows)
                for c in range(cols)
                if (r, c) not in (self.refuel, self.dropoff)
            )
        self.eligible_cells = eligible_cells

    # ------------------------------------------------------------------
    # Observation digest (the P1 input on active turns)
    # ------------------------------------------------------------------

    def observation(self, state: GasGridState) -> Observation:
        return (
            state.pos,
            gas_band(state.gas, self.gas_max),
            state.task_loc,
            state.carrying,
        )

    # ------------------------------------------------------------------
    # Environment primitives
    # ------------------------------------------------------------------

    @property
    def p1_alphabet(self) -> list[Action]:
        cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        task_locs: list[Cell | None] = [None, *self.eligible_cells]
        observations = [
            (pos, band, tl, carry)
            for pos in cells
            for band in GAS_BANDS
            for tl in task_locs
            for carry in (False, True)
        ]
        task_events = [("TASK", loc) for loc in self.eligible_cells]
        return [*task_events, *observations]

    def initial_state(self) -> GasGridState:
        return GasGridState(
            pos=self.home,
            gas=self.gas_max,
            task_loc=None,
            carrying=False,
            delivered=False,
            player=Player.P1,
            step_count=0,
        )

    def current_player(self, state: GasGridState) -> Player:
        return state.player

    def legal_actions(self, state: GasGridState) -> list[Action]:
        if self.is_terminal(state):
            return []
        if state.player is Player.P1:
            if state.task_loc is None:  # idle: P1 chooses which task arrives
                return [("TASK", loc) for loc in self.eligible_cells]
            return [self.observation(state)]  # active: forced observation reveal
        return self._p2_actions(state)

    def _p2_actions(self, state: GasGridState) -> list[Action]:
        actions: list[Action] = []
        r, c = state.pos
        for move, (dr, dc) in _MOVES.items():
            if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols:
                actions.append(move)
        if (
            state.task_loc is not None
            and state.pos == state.task_loc
            and (not state.carrying)
        ):
            actions.append(PICKUP)
        if state.pos == self.dropoff and state.carrying:
            actions.append(DROP)
        if state.pos == self.refuel:
            actions.append(REFUEL)
        return actions

    def step(self, state: GasGridState, action: Action) -> GasGridState:
        if state.player is Player.P1:
            if isinstance(action, tuple) and action and action[0] == "TASK":
                return state._replace(task_loc=action[1], player=Player.P2)
            return state._replace(player=Player.P2)  # observation reveal

        nxt = state._replace(player=Player.P1, step_count=state.step_count + 1)
        if action in _MOVES:
            dr, dc = _MOVES[action]
            return nxt._replace(
                pos=(state.pos[0] + dr, state.pos[1] + dc), gas=state.gas - 1
            )
        if action == PICKUP:
            return nxt._replace(carrying=True)
        if action == DROP:
            return nxt._replace(carrying=False, delivered=True)
        if action == REFUEL:
            return nxt._replace(gas=self.gas_max)
        raise ValueError(f"illegal P2 action {action!r}")

    def is_terminal(self, state: GasGridState) -> bool:
        if state.delivered or state.step_count >= self.max_steps:
            return True
        return self.gas_is_fatal and state.gas <= 0

    def reward(self, state: GasGridState) -> float:
        if state.delivered:
            return 1.0
        return -1.0  # gas-out (when fatal) or timeout without delivery


def gas_depleted(state: GasGridState) -> bool:
    """Safety predicate for ``G(gas > 0)`` — the target of the shielding layer."""
    return state.gas <= 0


def safety_state_key(state: GasGridState) -> tuple:
    """Abstraction for the gas spec: drop step_count/delivered (the spec ignores
    them), keeping the safety state space finite and patches state-keyed."""
    return (state.pos, state.gas, state.task_loc, state.carrying, state.player)


def manhattan_greedy_heuristic(env: GasGridEnv, state: GasGridState) -> float:
    """A deliberately *suboptimal* preference heuristic: rewards distance progress
    toward the goal but **ignores gas** — exactly the reward-hacking-prone policy
    the paper contrasts against. Carrying outranks not-yet-picked-up; both are
    scaled into ``(-1, 1)`` from P2's perspective."""
    if state.delivered:
        return 1.0
    if state.gas <= 0:
        return -1.0
    span = env.rows + env.cols
    if state.carrying:
        d = manhattan(state.pos, env.dropoff)
        return 0.5 + 0.5 * (1 - d / max(1, 2 * span))
    if state.task_loc is not None:
        d = manhattan(state.pos, state.task_loc) + manhattan(
            state.task_loc, env.dropoff
        )
        return 0.5 * (1 - d / max(1, 3 * span))
    return 0.0
