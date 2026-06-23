# PALS — Build Plan (camera-ready rebuild)

_Date: 2026-06-23. Follows `01_holistic_review.md`. Reflects locked decisions:_
- **Environments:** keep game-theoretic envs **and** add Gymnasium toy-text (Taxi, FrozenLake).
- **Liveness:** scope to **safety only**; correct the paper's claims. No Büchi.
- **Shielding backend:** **pure-Python** generic safety-game solver by default; Spot optional.

Testing discipline (per user): **unit tests for every function**, **end-to-end tests after**.
All test files `test_`-prefixed so pytest discovers them.

**Git workflow (per user):** proper GitHub flow on `github.com/cscaff/PALS` — each
phase / unit of work is a tracked **issue**, done on a **feature branch**, merged
via **PR** gated by CI (ruff + pytest). No feature commits directly to `main`.

**Stance toward the old repo (per user):** this is a **rebuild from the goals, not a
port**. The old code is a *reference for what worked*, not a template to recreate.
We reuse an old component only when re-deriving it from scratch would land in the
same place — and we rewrite freely (including the "clean" core) wherever a better
design serves the overall goals: symbolic policy via active automata learning,
preference-guided improvement over a suboptimal oracle via MCTS audit, PAC
equivalence, **generic safety shielding**, and a clean harness spanning
game-theoretic + toy-text envs with ablations + robustness. Do not get anchored on
old implementation details.

---

## Phase 0 — Scaffold
- `pals/` package layout from §5 of the review; `pyproject.toml`, pinned deps
  (z3-solver, gymnasium, pytest; Spot guarded behind an optional extra).
- **Style & tooling (PEP8, enforced):** all code PEP8. **ruff** as linter +
  formatter (config in `pyproject.toml`); **pytest** for tests. No hardcoded paths;
  Spot import wrapped in try/except.
- **CI:** GitHub Actions workflow running `ruff check` + `ruff format --check` +
  `pytest` on push and PR.
- **Exit:** `ruff check` and `pytest` both run green on an empty suite; CI passes;
  `import pals` works.

## Phase 1 — Environment protocol + first env
- `envs/base.py`: design a clean `Environment` ABC from first principles. The old
  repo's implicit protocol (`root`, `get_node(trace)`, `children`, `is_terminal`,
  `player`, `p1_alphabet`, `p2_legal_moves(trace)`) is one data point — adopt only
  the parts that earn their place; redesign the interface if a better one serves
  both game-theoretic and toy-text envs.
- Build **Nim** first (smallest, well-understood) against the new ABC.
- **Unit tests:** every board/NFA/oracle method. **Exit:** Nim conforms to ABC,
  100% of its functions unit-tested.

## Phase 2 — Core algorithm (rebuild)
- Rebuild `core/`: `lstar.py`, `sul.py` (collapse the dual prefix/state override
  paths into one coherent model), `table_b.py`, `smt.py`, `counting.py`. Reuse the
  old core's *ideas* (mutable-SUL L\*, targeted row invalidation) where sound;
  rewrite the structure freely.
- Rebuild `oracles/`: `mcts_audit.py` (clean rewrite — readable names, docstrings),
  `pac.py`, `composite.py`.
- **Unit tests** for every function — this is where the old repo was weakest
  (mcts_oracle had ~5 undiscovered tests). Target the SMT valuation, deviation
  sampling, UCB scoring, table invalidation explicitly.
- **e2e:** Nim learns a correct controller end-to-end; query counts recorded.
- **Exit:** PALS reproduces a Nim result; core coverage high.

## Phase 3 — Remaining envs
- Port TTT, Dots&Boxes, Minimax behind the ABC.
- Add **Gymnasium toy-text adapters**: Taxi (the paper's own motivation — currently
  unbenchmarked) and FrozenLake (natural safety spec). Adapter wraps a Gym env as
  an `Environment` with a tractable symbolic encoding.
- Per env: preference oracle + unit tests + a learning e2e test.
- **Exit:** all envs conform; each has a passing e2e learning test.

## Phase 4 — Generic shielding (the review's #1 priority)
- `shielding/safety_game.py`: extract & generalize the backward-attractor solver
  (env-agnostic; needs only `is_bad`, `children`, `player`, `is_terminal`).
- `shielding/spec.py`: spec abstraction (`is_bad` predicate; optional LTL string).
- `shielding/shield_oracle.py`: composite stage that patches the SUL from the
  safety strategy — rewritten without the STUCK/escalation hackery; clean CEX
  contract with L\*.
- `shielding/model_check.py`: pure-Python verifier default; **optional** Spot LTL
  backend behind the extra.
- **Deliver the active-shielding experiment:** ≥1 env (FrozenLake or Taxi-gas)
  where the unshielded hypothesis violates the spec and the shield demonstrably
  patches it. Capture before/after traces.
- **Unit tests** for solver/patcher/verifier; **e2e** for the shielded run.

## Phase 5 — Benchmark harness + baselines + ablations
- `bench/baselines.py`: shared Q-learning, UCT variants, greedy, optimal, random
  (de-duplicated from the ~30 per-game scripts).
- `bench/harness.py` + `configs/`: one config-driven runner; sweeps over
  `depth_n`, `K`, oracle-depth, env params.
- **Ablations as config flags** (CompositeEqOracle makes this cheap): MCTS audit
  on/off, PAC on/off, shield on/off.
- **Robustness study:** `NoisyOracle` wrapper; sweep preference-noise level.
- **Exit:** reproducible tables/plots; precise win/tie/loss reporting per env.

## Phase 6 — Paper-alignment pass
- Correct abstract/intro (safety-only; soften "first"; clarify Mealy-not-NFA and
  controller-vs-environment framing).
- Document each preference oracle + the new shielding & robustness results.

---

## Sequencing note
Phases 1–2 are the critical path (protocol + trustworthy core). Phase 4 (shielding)
is the highest-value *paper* deliverable and depends on the toy-text envs from
Phase 3 for its best demo. Start at Phase 0→1→2.
