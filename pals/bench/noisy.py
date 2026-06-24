"""Noisy preference oracle wrapper for robustness studies.

Wraps any :class:`~pals.core.preference.PreferenceOracle` and corrupts a
``noise`` fraction of its answers: a corrupted ``compare`` returns a fixed wrong
verdict and a corrupted ``preferred_move`` returns a fixed wrong legal move.

The corruption is **deterministic in the input** (seeded by the trace), not
re-randomized per call. This is essential: an oracle that answered differently
each time it was asked the same thing would make the system-under-learning
non-deterministic and L* would never converge. A fixed-but-imperfect oracle
models an *inconsistent teacher* — wrong on some inputs, but consistently so —
which is exactly the sensitivity setting the reviewers asked about.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

from pals.core.preference import Preference, PreferenceOracle
from pals.envs.base import Action, Environment


class NoisyOracle:
    def __init__(
        self,
        inner: PreferenceOracle,
        env: Environment,
        noise: float = 0.1,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= noise <= 1.0:
            raise ValueError("noise must be in [0, 1]")
        self.inner = inner
        self.env = env
        self.noise = noise
        self.seed = seed

    def _rng_for(self, key: object) -> random.Random:
        """A deterministic RNG keyed by the query, so the corruption is a fixed
        function of the input rather than per-call randomness. Seeded via a
        stable hash (process-independent, unlike ``hash`` on tuples)."""
        digest = hashlib.sha256(repr((self.seed, key)).encode()).hexdigest()
        return random.Random(int(digest[:16], 16))

    def preferred_move(self, trace: Sequence[Action]) -> Action | None:
        rng = self._rng_for(("move", tuple(trace)))
        if rng.random() < self.noise:
            legal = self.env.p2_legal_moves(trace)
            if legal:
                return rng.choice(legal)
        return self.inner.preferred_move(trace)

    def compare(self, trace1: Sequence[Action], trace2: Sequence[Action]) -> Preference:
        rng = self._rng_for(("compare", tuple(trace1), tuple(trace2)))
        if rng.random() < self.noise:
            return rng.choice(list(Preference))
        return self.inner.compare(trace1, trace2)
