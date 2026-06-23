"""The shielding equivalence-oracle stage.

Enforces a safety spec on the learned controller. At construction it solves the
safety game once (the safe strategy). On each ``find_counterexample`` it model-
checks the current hypothesis: if the controller can reach a bad state, it
installs **state-keyed** safety patches on the offending P2 states (so the SUL
emits the safe action there, on every trace reaching that state) and returns the
violating trace so L* relearns the now-safe behaviour. When no violation remains
the controller provably satisfies the spec and the shield reports convergence.

As a composite stage placed after the MCTS audit and PAC, this realizes the
paper's design: recover preference-optimal play where the spec permits, and fall
back to a safe action only where it would otherwise be violated.
"""

from __future__ import annotations

from collections.abc import Callable

from pals.core.lstar import MealyMachine
from pals.core.sul import PreferenceSUL
from pals.envs.base import Action, Environment, Player, State
from pals.shielding.model_check import find_violation
from pals.shielding.safety_game import solve_safety_game
from pals.shielding.spec import SafetySpec


class ShieldOracle:
    def __init__(
        self,
        env: Environment,
        sul: PreferenceSUL,
        spec: SafetySpec,
        prefer_action: Callable[[State], Action | None] | None = None,
        verbose: bool = False,
    ) -> None:
        self.env = env
        self.sul = sul
        self.spec = spec
        self.verbose = verbose
        self._key = spec.state_key or (lambda s: s)

        self.game = solve_safety_game(
            env, spec.is_bad, prefer_action=prefer_action, state_key=spec.state_key
        )
        # Route the SUL's safety patches through the same state abstraction.
        sul.state_key_fn = self._key

        self.num_queries = 0
        self.patches_installed = 0

    def find_counterexample(self, hypothesis: MealyMachine) -> list[Action] | None:
        self.num_queries += 1
        violation = find_violation(self.env, hypothesis, self.spec.is_bad)
        if violation is None:
            return None

        # Patch the unsafe P2 choices along the violating trace.
        installed = self._patch_along(violation)
        if self.verbose:
            print(f"  [shield] violation len={len(violation)}, patched={installed}")
        if installed == 0:
            # No safe action to install (spec unrealizable from here, or already
            # patched): we cannot make progress on this violation.
            return None
        self.patches_installed += installed
        return self.sul.p1_inputs_of(violation)

    def _patch_along(self, violation: list[Action]) -> int:
        """Install state-keyed safe patches at the P2 nodes of the violation
        whose safe action differs from the action taken. Returns how many new
        patches were installed."""
        env = self.env
        state = env.initial_state()
        installed = 0
        for action in violation:
            if env.is_terminal(state):
                break
            if env.current_player(state) is Player.P2:
                key = self._key(state)
                safe = self.game.safe_action(key)
                if (
                    safe is not None
                    and safe != action
                    and self.sul._state_overrides.get(key) != safe
                ):
                    self.sul.patch_state(key, safe)
                    installed += 1
            state = env.step(state, action)
        return installed
