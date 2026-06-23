"""Safety specifications.

A :class:`SafetySpec` is just a state predicate marking spec violations, plus
optional metadata. ``G(gas > 0)`` becomes ``SafetySpec(lambda s: s.gas <= 0)``.
``state_key`` optionally quotients the state space for the safety analysis (must
be a sound abstraction for the spec — e.g. dropping a step counter the spec
ignores). The predicate is the single domain-specific input the shield needs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pals.envs.base import State


@dataclass(frozen=True)
class SafetySpec:
    is_bad: Callable[[State], bool]
    name: str = "G(safe)"
    ltl: str | None = None
    state_key: Callable[[State], object] | None = None
