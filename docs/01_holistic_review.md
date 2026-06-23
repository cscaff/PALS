# PALS — Holistic Review (Paper · Old Repo · Reviews)

_Date: 2026-06-23. Purpose: ground the from-scratch rebuild. Maps what exists →
what the paper claims → what the reviews demand → what we should build._

---

## 1. The algorithm, in one paragraph

PALS learns a **deterministic Mealy machine** (a symbolic policy for the "system"
player P2) by running a custom **L\*** loop whose System-Under-Learning is a pair
`(environment NFA, preference oracle)`. The environment (P1) is *enumerated, not
optimized* — every legal P1 input is treated as equally valid. Membership queries
return the oracle's locally-greedy P2 response at a prefix. The equivalence oracle
is a **composite** of three stages: (1) an **MCTS audit** that rolls out to depth
N from sampled deviation points, uses an SMT solver to turn pairwise preferences
into numeric leaf values, and — if a majority of rollouts prefer a deviation —
*mutates the oracle's strategy* and returns the deviation as a counterexample;
(2) a **PAC** check that samples random walks and compares hypothesis vs SUL;
(3) a **safety/shielding** stage that model-checks an LTL spec and patches the SUL
toward safe actions. The CEX drives L\* to refine. It terminates at a depth-N
locally-optimal, spec-respecting controller.

---

## 2. Old-repo architecture map

Location: `.../Spring_26/COMS 4232/Final Project/Imperfect_Information_Automata_Learning`,
branch `Scaff_Game_Implementations`. ~13.7k LoC of `src`.

### Core algorithm — `src/lstar_mcts/` — **KEEP (refactor lightly)**
This part is genuinely clean and already game-agnostic:

| File | Role | Verdict |
|---|---|---|
| `learner.py` | `run_lstar_mcts()` wiring entry point | Keep |
| `custom_lstar.py` | Mealy L\* with targeted row invalidation (lets SUL mutate mid-run) | Keep — this is the real novelty vs AALpy |
| `mcts_oracle.py` | MCTS audit stage; deviation sampling, SMT valuation, strategy update | Keep, but **under-tested & hard to read** (camelCase methods, dense) |
| `pac_oracle.py` | Kearns-Vazirani PAC sample schedule | Keep |
| `composite_oracle.py` | Variadic chain of eq-oracle stages | Keep — this is what makes ablations trivial |
| `table_b.py` | UCB exploration table, persists across rounds | Keep |
| `game_sul.py` | SUL with prefix- and state-keyed overrides + caching | Keep, simplify the dual override paths |
| `smt_solver.py` | Z3 ordinal→numeric value assigner | Keep |
| `counting_oracle.py` | query-cost instrumentation wrapper | Keep |

The games all implement the **same implicit protocol** (`root`, `get_node(trace)`,
`children`, `is_terminal()`, `player`, `p2_legal_moves(trace)`, `p1_alphabet`).
This should become an explicit `Environment` ABC in the new build.

### Shielding — `src/control_systems/RobotGrid/safety_oracle.py` (793 LoC) — **REWRITE**
The user's instinct is correct: this is the mess. Problems:
- **Game-specific.** Hardwired to `RobotGridState` and the gas spec. Nothing here
  works for any other environment.
- **Hardcoded dependency path:** `/opt/homebrew/Cellar/spot/2.14.1/.../site-packages`
  is injected into `sys.path`. Brittle, non-portable.
- **Four concerns tangled in one file:** spec predicate, Spot LTL model-checking,
  backward-attractor safety-game synthesis, and CEX patching.
- **Fragile L\* integration:** large amounts of "trimmed → full-escalated → STUCK"
  CEX-dedup machinery (`_returned_cexes`, `_stuck_announced`) — a symptom that
  feeding safety CEXes back into L\* doesn't cleanly converge.
- **The good part:** `solve_safety_game()` (backward attractor over reachable
  abstract states) is actually generic in spirit — it only needs `is_bad`,
  `children`, `player`, `is_terminal`. Extract and generalize it. The pure-Python
  safety-game solver can replace Spot entirely for safety specs.

### Games — `src/game/{nim,tic_tac_toe,dots_and_boxes,hex,minimax,grid_nav}` — **PORT**
Each = `board.py` + `game_nfa.py` + `preference_oracle.py`. Preference oracles are
**depth-limited minimax over a hand-coded greedy heuristic** (e.g. Nim uses
largest-pile fraction, deliberately ignoring nim-sum so it's locally-but-not-
globally optimal). Well-structured; port behind the new `Environment` ABC.

### Eval/baselines — `src/scripts/` + `src/eval/` — **CONSOLIDATE**
~30 near-duplicate scripts (`learner_X.py`, `eval_X.py`, `mcts_player_X.py`,
`mcts_trained_player_X.py`, `aggregate_X.py` per game). Baselines (Q-learning, UCT
variants, greedy, optimal, random) are reimplemented per game. Collapse into one
config-driven benchmark harness + shared baseline implementations.

### Tests — 405 test functions, but **core is barely covered**
Game boards are well tested. **Critical gap:** the core-algorithm test files are
**not `test_`-prefixed**, so pytest never discovers them:
`tests/lstar_mcts/{game_sul,mcts_oracle,smt_solver}.py`,
`tests/game/minimax/*.py`. `mcts_oracle.py` — the actual PALS contribution — has
only ~5 tests, undiscovered. The thing most needing tests has the least.

---

## 3. Paper claims vs. reality (the review fault-lines)

| Claim in paper | Reality in code | Review flagging it |
|---|---|---|
| Handles **safety AND liveness** (abstract/intro) | Safety only; §3.3 admits liveness needs Büchi, "future work" | R1 (major), R3 |
| Shielding layer is a **core component** | In results, "no safety specs imposed, shielding inactive." Only RobotGrid wires it, and RobotGrid **isn't in the paper's results** | R1 (killer point) |
| "**Better than or comparable to strongest baselines**" | Minimax table: PALS `vs_opt 57.67` vs Optimal `59.78`, UCT_terminal `57.50` — a *tie*, not a win | R1, R2 |
| "**First** to fully symbolically learn RL policies via automata learning" | Overclaim; needs softening vs Neider, Kouteili, Remap | R1 |
| Learns an **NFA** | Learns a deterministic **Mealy machine**; nondeterminism is only in the environment's output relation | R2 (NFA-learning hardness) |
| Taxi Driver is the motivating example (Fig 1) | Taxi/`grid_nav`/`RobotGrid` exists in code but is **not benchmarked** in the paper | implicit gap |
| Preference oracle "encodes user notions of good play" | Hand-coded depth-limited minimax heuristics; **no noise/robustness study** | R1 (how implemented? sensitivity?) |

R2's deeper point: these are **controller-vs-enumerated-environment** problems, not
multi-agent equilibrium settings. The paper's "game-theoretic" framing invites an
equilibrium reading it doesn't deliver. Either reframe honestly as symbolic
controller synthesis, or lean into the Gymnasium toy-text framing (where "agent vs
environment" is the native and uncontested model).

---

## 4. What the reviews concretely demand (actionable backlog)

1. **An active-shielding experiment.** ≥1 benchmark where the learned hypothesis
   violates an LTL safety spec and the shield patches it. _Highest priority — it
   neutralizes R1's strongest objection._
2. **Fix the liveness claim.** Either implement Büchi acceptance or scope claims to
   safety. (Recommend scope-down; Büchi is a large, separate lift.)
3. **Specify the preference oracle** per benchmark + a **noise/robustness** study.
4. **Precise empirical comparison** — say exactly where PALS wins/ties/loses.
5. **Ablations:** MCTS audit on/off, PAC on/off, shield on/off. (`CompositeEqOracle`
   makes this nearly free.)
6. **Soften the "first" claim**; expand related work.
7. **Qualify the proof** as conditional on Assumptions 1–2; say when they hold.
8. Clarify game class (zero-sum? stochastic? imperfect-info?) and notation.

---

## 5. Recommended new architecture

```
pals/
  core/            # game-agnostic algorithm (ported from lstar_mcts/)
    lstar.py            # Mealy L* w/ targeted invalidation
    sul.py              # GameSUL, simplified override model
    table_b.py, smt.py, counting.py
  oracles/
    mcts_audit.py       # the MCTS deviation oracle (readable rewrite)
    pac.py
    composite.py
  shielding/         # NEW: fully generic, env-agnostic
    spec.py             # LTL formula / is_bad predicate abstraction
    safety_game.py      # generic backward-attractor solver (extracted)
    shield_oracle.py    # composite stage; patches SUL from safety strategy
    model_check.py      # optional Spot backend; pure-Python fallback default
  envs/
    base.py             # Environment ABC (the implicit protocol, made explicit)
    nim.py, tic_tac_toe.py, dots_and_boxes.py, minimax.py, ...
    gym_toytext/        # Taxi, FrozenLake, CliffWalking adapters
  oracles_pref/        # preference oracles + NoisyOracle wrapper
  bench/
    harness.py          # one config-driven runner
    baselines.py        # shared Q-learning, UCT variants, greedy, optimal, random
    configs/            # per-experiment YAML/JSON
tests/                 # test_-prefixed, one file per module, unit + e2e
```

Guiding principles: keep the clean core; make the `Environment` protocol explicit
so a new env is one file; make shielding **generic and Spot-optional**; make
ablations config flags, not forks.

---

## 6. Open decisions (need user input)

1. **Environment scope** — more Gymnasium toy-text (Taxi/FrozenLake; supports the
   shielding story and the paper's own motivation) vs more game-theoretic vs both?
2. **Liveness** — implement Büchi, or scope to safety and correct the paper?
3. **Shielding backend** — keep Spot (heavy, brittle path) or default to the
   pure-Python safety-game solver and make Spot an optional backend?
