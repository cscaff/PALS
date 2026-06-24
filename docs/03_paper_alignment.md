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

**Minimax (depth 6, b=2)** — *clean win.* On a non-degenerate depth-6 tree PALS
reaches `89.9 / 81.4 / 80.3`, **matching exact Optimal** (`90.2 / 81.4 / 80.3`) on
the strong-opponent columns and **beating UCT** (`84.1 / 70.3 / 64.2`) and Greedy
(`81.4 / 56.8 / 56.8`). The ablation is decisive: **PALS_no_mcts collapses to the
greedy level** (`56.8 / 56.8` on vs_greedy/vs_optimal) — the audit is what closes
the gap to optimal. (The earlier depth-4 instance was degenerate: many strategies
tied at one value because the tree was too small to separate them.)

**Tic-Tac-Toe** — *optimal-equivalent on the hard columns.* PALS forces the draw
that perfect play guarantees vs both greedy and optimal P1 (`0.000 / 0.000`),
exactly like Optimal, while **UCT and Q-learning lose** to the optimal opponent
(`UCT −0.93 / −0.83`, `QLearning −1.00 / −1.00`). PALS is slightly behind Greedy
vs a random opponent (`0.667` vs `0.833`) but never loses to strong play.

**Dots & Boxes (2×2)** — *competitive, honestly stated.* On the non-degenerate
2×2 board PALS scores `0.900 / 0.000 / 0.000`: it ties Greedy and beats UCT vs the
optimal opponent (`UCT −0.067`), but exact Optimal wins outright (`1.000 / 1.000 /
1.000`). State this as "above Greedy/UCT, below exact Optimal," not "beats." (The
earlier 1×2 board was degenerate — every P2 lost to optimal P1.)

Reporting rule: say where PALS **wins** (Nim, Minimax), is **optimal-equivalent**
where the game forces a draw (Tic-Tac-Toe), and is **competitive but below exact
Optimal** (Dots & Boxes 2×2), per R1's explicit request.

**Figures** (`docs/results/figures/`, via `python -m scripts.plot_benchmarks`,
needs the `viz` extra): per-game grouped bar charts (`nim.png`, `minimax.png`,
`tic_tac_toe.png`, `dots_and_boxes.png`) and a rollout-budget sweep
(`minimax_k_sweep.png`) showing
PALS improving with K and then saturating — the honest version of the paper's
Fig 5. Note: the original Fig 5's monotone-in-K claim only holds on deeper trees
with a larger audit depth N; on shallow instances PALS saturates immediately.

---

## 3. The active-shielding experiment (R1's #1 ask)

Setup: gas corridor (1×3), spec `G(gas > 0)`, with `gas_is_fatal=False` so the
gas-blind preference oracle's objective is **misaligned** with safety — the
genuine reward-hacking regime where preference optimization alone cannot fix
safety. Result (`docs/results/benchmarks.txt`):

- **Unshielded:** a reachable `G(gas>0)` violation exists (the controller drives
  the tank to zero on the fast path).
- **Shielded (with `prefer_action`):** **no** reachable violation, exactly **1**
  state-keyed safe patch, and the controller **still delivers** (reward 1.0):
  `PICKUP → E → REFUEL → E → DROP` — one refuel inserted to stay safe.

This is the demonstration the paper was missing. Method: a pure-Python backward-
attractor safety-game solver + reachability model checker (no Spot dependency);
the shield patches the SUL toward safe actions only where the spec would be
violated, recovering preference-optimal play elsewhere. See
`docs/results/figures/shielding.png`.

**Design point worth stating:** pure safety shielding can produce *safe-but-
useless* controllers — if the safe strategy picks an arbitrary safe action it may
abandon the task (e.g. refuel forever). Passing the preference as a tie-breaker
(`prefer_action`) keeps the controller's preferred move wherever it is safe, so
the shielded controller is both safe **and** task-completing. This is exactly the
paper's "recover preference-optimal play where the spec permits" claim, and the
codebase now demonstrates it concretely.

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
  The sweep is reproducible — see the two "Noise sensitivity" tables in
  `docs/results/benchmarks.txt` (`python -m scripts.run_benchmarks`, seed 0;
  noise ∈ {0, 0.1, 0.25, 0.5} on Nim and Minimax, full PALS vs the audit-off
  `PALS_no_mcts` core).

  Measured finding (state precisely; the earlier "audit fails to terminate"
  phrasing was too strong — it does terminate, but degrades):
  - The **L\*+PAC core (`PALS_no_mcts`) is robust**: automaton size stays small
    (5–8 states) and play quality is stable as noise grows — on Minimax it holds
    `81.4 / 80.3` on vs_greedy/vs_optimal at every noise level including 0.5.
  - The **MCTS audit degrades under inconsistent preferences**: accepted
    deviations explode (Minimax `1 → 217` at noise 0.1, `39` at 0.25), and at
    **noise 0.5 the full PALS is *worse* than its audit-off core** on vs_optimal
    (`44.8` vs `80.3`) — the audit chases noise-induced "deviations" that do not
    reflect real strategic gains.
  - On **Tic-Tac-Toe** (large alphabet) the same effect is dramatic: the audit's
    deviation count and automaton blow up with noise (states `76 → 257`,
    deviations `0 → 218`, runtime `0.4 s → 286 s` at noise 0.5), which is why TTT
    is kept out of the default fast sweep and documented here instead.
  - **Take-away for the paper:** Theorem 2's termination/optimality assumes
    consistent, history-independent preferences (Assumption 1). Under a noisy
    teacher the audit still halts (via L*'s round cap) but loses its guarantee and
    can hurt; the L\*+PAC path remains a robust fallback. Frame this as the
    practical scoping condition on Theorem 2, with the tables as evidence.

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
