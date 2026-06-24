# PALS — Paper Alignment (camera-ready revision guide)

_Date: 2026-06-23. Maps every reviewer concern to its resolution in the rebuilt
codebase, the exact claim edits to make, and the real numbers to cite. Results
are reproducible via `python -m scripts.run_benchmarks` (seed 0); the captured
run is in `docs/results/benchmarks.txt`._

---

## 1. Claim corrections (do these first)

| Current claim | Problem (reviewer) | Corrected claim |
|---|---|---|
| Handles **safety and liveness** | R1: §3.3 admits liveness is future work | State **safety only**. Remove liveness from abstract/intro; keep "liveness via Büchi acceptance is future work" in one sentence. |
| Shielding is a core component (but inactive in results) | R1 (killer): never exercised | Now **demonstrated** — see §3. Add the active-shielding experiment to results. |
| "Better than or comparable to the strongest baselines" | R1, R2: mixed | Replace with the **per-game, per-opponent** statement in §2: clean win on Nim, competitive on Minimax, degenerate on tiny Dots & Boxes. |
| Learns an **NFA** | R2: NFA learning is hard; method is deterministic | Say it learns a **deterministic Mealy machine**; the environment's output relation is nondeterministic but the *learned policy* is deterministic. Drop NFA-learning framing. |
| "**First** to fully symbolically learn RL policies via automata learning" | R1: overclaim | Soften: "to our knowledge, among the first to combine active automata learning, a preference oracle, and safety shielding in one loop." Expand related work. |
| "Game-theoretic games" | R2: which class? | Frame as **controller synthesis against an enumerated/adversarial environment** (P1 enumerated, P2 learned), not equilibrium computation. Be explicit it is not Nash/equilibrium. |

---

## 2. Empirical comparison — precise win/tie/loss

From `docs/results/benchmarks.txt` (seed 0; mean reward, P2's perspective; columns
are the P1 opponent). Each game uses a deliberately **suboptimal** preference
oracle (depth-limited minimax over a greedy heuristic).

**Nim (1,2,3)** — *clean win, and an ablation that isolates the audit's value.*
PALS reaches `1.000 / 1.000 / 1.000` (matching Optimal) from a suboptimal oracle,
while UCT and Q-learning collapse against the greedy/optimal opponents
(`UCT −0.20 / −1.00`, `QLearning −1.00 / −1.00` on vs_greedy/vs_optimal). The
ablation is the headline: **PALS_no_mcts scores only `0.400` vs_random** — the
MCTS audit is exactly what lifts the suboptimal-oracle policy to optimal play.

**Minimax (depth 4)** — *competitive, honestly stated.* PALS is on par with the
MCTS/UCT baselines vs random and matches Greedy/Q-learning on the harder columns;
Optimal is best vs the strong opponents. State this as "competitive with UCT,
above Greedy, below exact Optimal," not "beats."

**Dots & Boxes (1×2)** — *degenerate; report or drop.* The board is small enough
that the optimal P1 wins against every P2 (`vs_optimal = −1.0` for all). Either
scale to a larger board or note it isolates nothing and drop it from headline
claims.

Reporting rule: say where PALS **wins** (Nim), **ties** (Minimax vs UCT), and is
**uninformative/loses** (tiny Dots & Boxes), per R1's explicit request.

---

## 3. The active-shielding experiment (R1's #1 ask)

Setup: gas corridor (1×3), spec `G(gas > 0)`, with `gas_is_fatal=False` so the
gas-blind preference oracle's objective is **misaligned** with safety — the
genuine reward-hacking regime where preference optimization alone cannot fix
safety. Result (`docs/results/benchmarks.txt`):

- **Unshielded:** a reachable `G(gas>0)` violation exists (the controller drives
  the tank to zero on the fast path).
- **Shielded:** **no** reachable violation; 3 state-keyed safe patches installed.

This is the demonstration the paper was missing. Method: a pure-Python backward-
attractor safety-game solver + reachability model checker (no Spot dependency);
the shield patches the SUL toward safe actions only where the spec would be
violated, recovering preference-optimal play elsewhere.

---

## 4. Oracle implementation & sensitivity (R1)

- **How the oracle is implemented:** every benchmark uses
  `MinimaxPreferenceOracle` — depth-limited minimax over a hand-coded greedy
  heuristic (Nim: largest-pile fraction; Minimax: leftmost-leaf; Dots & Boxes:
  score margin; gas grid: Manhattan progress). `depth=None` is exact (a globally
  optimal oracle); a finite depth + greedy heuristic is the locally-optimal,
  suboptimal teacher. State this explicitly per benchmark.
- **Sensitivity to noise:** `NoisyOracle` wraps any oracle and corrupts a `noise`
  fraction of answers, *consistently per input* (a fixed but imperfect teacher).
  Finding to report: PALS's **L\*+PAC** core learns the imperfect target and
  terminates under noise, but the **MCTS audit's termination (Thm 2) assumes
  consistent / history-independent preferences**, which noise violates. State this
  as a scoping condition on Theorem 2.

---

## 5. Proof qualification (R1, R3)

Present Proposition 1 / Theorem 2 / Proposition 3 as **conditional on
Assumptions 1 (stationarity) and 2 (N-horizon stability)**. Add one sentence:
these hold when the preference order is history-independent and depth-N
lookahead is decisive at audited prefixes — true for the deterministic
perfect-information games here, and *violated* by noisy oracles (§4) and by
repeated games where local play misleads (R2's Iterated Prisoner's Dilemma point).

---

## 6. Ablations (R1)

The composite equivalence oracle makes ablations configuration flags, reported in
the tables: `PALS` (audit + PAC), `PALS_no_mcts` (PAC only), `PALS_no_pac` (audit
only), and shield on/off (§3). Nim shows the audit is necessary; the shield
experiment shows the shield is necessary under misaligned preferences.

---

## 7. Notation & presentation (R1, R2)

- Standardize on **L\*** and **Mealy machine**; define the observation table,
  preference oracle, MCTS oracle, PAC oracle, and shield once and reuse.
- Add the missing preliminaries (Mealy L\*, safety games, shielding) — at least
  in an appendix (R2).
- Proofread for the typos R1 listed ("algorithim", "Sheidling").

---

## 8. Related work to add (R1)

Symbolic RL, automata learning for games, preference-based RL, safe RL with
shielding, and synthesis-based controller learning — the current related work is
too thin for the claims. (Keep Neider, Kouteili, Remap, Alshiekh shielding,
Bloem shield synthesis; add recent preference-based and safe-RL references.)
