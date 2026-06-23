# Guide

I am currently working on building from scratch a production-ready version of past experiments that were done. The paper detailing the experiments that were done "PALS: Preference-guided Active Learning for Symbolic RL PALS: Preference-guided Active Automata Learning for Symbolic Reinforcement Learning in Games" exists in /docs. We want to take this and prepare camera-ready experiments.

Our paper at https://openreview.net/forum?id=jRUJpASPQJ received the following reviews:

## Reviews

### Review 1.

Promising Symbolic RL Direction, but Core Claims Need Stronger Empirical and Formal Support
Official Review by Reviewer 1GVx23 May 2026 at 20:25 (modified: 25 May 2026 at 08:02)Program Chairs, Area Chairs, Reviewers, AuthorsRevisions
Review:
This paper introduces PALS, a preference-guided active automata learning framework for learning symbolic policies in games. The method combines an L-style learner, a preference oracle, MCTS-based auditing, PAC equivalence checking, and an LTL shielding layer. The goal is to learn inspectable and verifiable symbolic policies that can improve over a suboptimal preference oracle while respecting safety constraints. The paper motivates the approach using Taxi Driver and evaluates it on Minimax, Tic-Tac-Toe, Nim, and Dots and Boxes against Q-learning and MCTS variants.

The topic is relevant to NExT-Game because it connects symbolic reinforcement learning, game-theoretic benchmarks, MCTS, safety specifications, and verifiable policy learning. However, I do not think the paper is ready for acceptance in its current form because several central claims are not fully supported by the algorithm description or experiments.

Strengths:

The paper fits the NExT-Game theme well through symbolic RL, game-theoretic learning, MCTS, formal verification, and safety-aware policy learning.
Interesting research direction: Combining active automata learning with preference-guided policy learning and MCTS-based oracle refinement is a promising idea.
Symbolic policy learning is valuable.
The taxi example is good. It shows nicely the gap between reward-driven RL, correctness-only synthesis, and preference-guided safe behavior.
MCTS audit stage is conceptually useful.
Weaknesses:

The safety or liveness claims are inconsistent: The abstract and introduction suggest that PALS handles both safety and liveness objectives, but Section 3.3 states that the current method considers safety specifications and that extension to liveness requires Buchi acceptance, which is left to future work. This is a major claim-level inconsistency.
The shielding layer is not empirically evaluated. The paper presents the LTL shielding layer as a key part of the framework, but the results section states that no safety specifications are imposed and the shielding layer is inactive. This clearly demonstrates that the paper’s safety-enforcement claim is not actually tested.
The empirical evidence is mixed: The main text claims that PALS performs better than or comparably to the strongest baselines, but the appendix tables show mixed results across Nim, Tic-Tac-Toe, Dots and Boxes, and Minimax. In several cases, PALS does not clearly outperform Q-learning, optimal play, or MCTS variants.
The proof sketch depends on strong assumptions.
The paper says the oracle encodes user-supplied notions of good play, but it is not fully clear how the oracle is implemented in each benchmark, how realistic it is, or how robust PALS is to noisy/inconsistent preferences.
The claim that PALS is the first algorithm to fully symbolically learn RL policies for agents in games via automata learning should be softened unless the authors provide a more complete comparison with prior automata-learning, synthesis, shielding, and symbolic-control work.
Presentation needs cleanup. There are multiple typos and clarity issues, such as “algorithim,” “Sheidling,” and inconsistent terminology around L*, “L* algorithm,” and PALS. These issues are making the technical contribution harder to follow.
The idea is promising and relevant to NExT-Game, but the current paper overclaims relative to the evidence. Thus, I recommend "Marginally below acceptance threshold" for this paper. And major revisions are needed.

If the paper has to be accepted, then before final acceptance of the paper, I strongly recommend that the authors should address the following points, revise them thoroughly, and resubmit.

Major Revisions Needed:

Does the current version of PALS actually support liveness objectives, or only safety specifications?
Please revise the abstract, introduction, and contribution claims to match the method. If liveness is future work, it should not be presented as a current capability.

Can the authors provide an experiment where the shielding layer is active? Since shielding is one of the central components, the paper should include at least one benchmark where the learned hypothesis violates an LTL safety specification and the shielding layer patches the preference oracle.
How is the preference oracle implemented in each benchmark?
Please specify whether it is hand-coded, heuristic-based, learned, or user-specified, and discuss sensitivity to noisy or inconsistent preferences.

Please make the empirical comparison more precise. Clearly state where PALS outperforms, matches, or underperforms Q-learning, MCTS, greedy, and optimal baselines. The current broad claim of outperforming or matching the strongest baselines is not fully supported.
Please qualify the proof sketch. Clearly state them as conditional on the two assumptions. The paper should also explain when these assumptions are likely to hold in practical games.
If possible, can you please add ablations?
Add points separately about the evaluation of the contribution of the MCTS audit stage, the pAC equivalence check, and the shielding layer.

Use standardized notation. Clarify how the observation tables, preference oracle, MCTS oracle, PAC oracle, and shielding layer interact.
Please thoroughly proof read the revised paper before resubmitting as there are multiple typos, grammar errors, and punctuation marks errors.
References: The related work feels too sparse for the strength of the claims. Can you please add most recent strong references? I recommend adding references that supports more stronger discussion of symbolic RL, automata learning for games, preference-based RL, safe RL with shielding, and synthesis-based controller learning.
Thank you.

Rating: 5: Marginally below acceptance threshold
Confidence: 5: The reviewer is absolutely certain that the evaluation is correct and very familiar with the relevant literature

### Review 2.

Interesting area of work, but not entirely sure of the game-theoretic relevance.
Official Reviewby Reviewer aS5t23 May 2026 at 11:44 (modified: 25 May 2026 at 08:02)Program Chairs, Area Chairs, Reviewers, AuthorsRevisions
Review:
Summary:
The authors propose PALS (Preference-guided Active automata Learning for Symbolic Reinforcement Learning), an automata-learning framework for learning fully symbolic policies for goal-directed games from a preference oracle together with LTL (Linear Temporal Logic) safety specifications. The method combines an (L^*)-style active automata learner with an MCTS-based audit stage and a shielding layer enforcing safety constraints. The framework iteratively refines both the learned automaton and the preference oracle: MCTS rollouts search for strategically preferable deviations to the current hypothesis, while the shielding layer patches behaviors violating the safety specification. The authors provide a proof sketch of convergence to globally preferred behavior under assumptions on the preference structure and rollout depth, and experimentally evaluate the method on several benchmark games including Tic-Tac-Toe, Nim, Dots-and-Boxes, Minimax tree games, and the Taxi benchmark.

Strengths:
The paper studies an interesting direction at the intersection of automata learning, symbolic RL, and game-playing, where certain strategic preferences too can be encoded with LTL and ultimately is "learnable" in their framework.
The framework works under weaker locally-optimal preference oracles, rather than assuming access to globally optimal ones, and uses MCTS audits to iteratively refine the learned controller.
The use of LTL safety specifications through a shielding layer provides a principled way to enforce safety constraints while still optimizing preferential behavior.
The paper provides a proof sketch establishing convergence to globally preferred behavior under assumptions on the game structure and rollout depth.
The experiments on games such as Tic-Tac-Toe, Nim, Dots-and-Boxes, and random Minimax trees suggest that the approach can outperform or match several MCTS variants and Q-learning baselines under suboptimal preference oracles.
Weaknesses:
The paper does not introduce several important preliminaries (nor defer them to an appendix), which makes it harder for a general algorithmic game-theory audience to appreciate the technical contributions and assumptions.
It is not fully clear what precise class of “game-theoretic games” the results are intended to apply to. The experiments mainly involve deterministic perfect-information games (e.g., Tic-Tac-Toe, Nim, Minimax trees), but the paper does not clearly discuss whether the framework is intended for zero-sum, general-sum, stochastic, or imperfect-information settings.
As written, the framework appears conceptually closer to symbolic single-agent RL or controller synthesis augmented with adversarial search, and it is not fully clear how the learned policies correspond to standard game-theoretic equilibrium notions (e.g., Nash equilibrium or minimax optimality), nor what the key strategic challenges are from a multi-agent game-theoretic perspective.
The correctness discussion explicitly excludes settings such as Iterated Prisoner’s Dilemma where locally optimal play may fail to reveal globally optimal long-term strategic behavior. In my view, these repeated-game settings are among the most interesting strategically, since local preferences may be fundamentally misleading there.
The computational challenges of learning NFAs under evolving local preference-oracle assumptions are not discussed in much detail. In particular, active learning of NFAs is significantly more challenging than DFA learning in general, and the implications of nondeterminism for scalability and identifiability are not carefully addressed.
Rating: 4: Ok but not good enough - rejection
Confidence: 3: The reviewer is fairly confident that the evaluation is correct

### Review 3.

A Technically Sound Framework Bridging Formal Methods and Reinforcement Learning via Active Automata Learning and Preference-Guided Safety Shields
Official Review
Reply type
by Reviewer sUQ422 May 2026 at 13:49 (modified: 25 May 2026 at 08:02)Program Chairs, Area Chairs, Reviewers, AuthorsRevisions
Review:
The authors introduce PALS, an active framework that learns fully symbolic policies for goal-directed games. Rather than relying on traditional, heavily engineered reward functions, PALS builds policies entirely as verifiable finite-state machines. It accomplishes this by extending the classical  algorithm so that both the policy hypothesis and an underlying preference oracle co-evolve using Monte Carlo Tree Search (MCTS) exploration and Linear Temporal Logic (LTL) safety shields.

The major strength of this work lies in its ability to combine behavioral correctness with strategic optimality without the scalability bottlenecks of full reactive synthesis. PALS provides a sound mechanism for escaping local optima during exploration. Furthermore, the approach is supported by meaningful empirical results across various game-theoretic benchmarks, where it performs competitively against traditional reinforcement learning baselines.

However, the framework exhibits a few weaknesses. First, its theoretical convergence is showcased under restrictive assumptions such as a perfectly stationary environment and the requirement that optimal paths remain within a fixed rollout depth. Finally, despite scaling better than complete formal synthesis, maintaining and expanding tables and invoking solvers could still face severe computational bottlenecks in games with massive action spaces.

Overall, this is a clear accept for a workshop venue. It presents an interesting and technically sound bridge between formal methods and reinforcement learning, delivering interpretable symbolic agents with promising experimental results.

Rating: 8: Top 50% of accepted papers, clear accept
Confidence: 3: The reviewer is fairly confident that the evaluation is correct

## Current Repository

The current repository lives at:

/Users/christianscaff/Documents/Academics/Columbia/Courses/Spring_26/COMS 4232/Final Project/Imperfect_Information_Automata_Learning

The latest branch we worked on was branch-"Scaff_Game_Implemnetations". It was not merged to main but remains the main branch we used for the paper.

## Goal

Given the state of the latest branch, the paper, and the reviews, I want to start from scratch and develop a production-quality codebase here for conducting camera-ready experiments. That means getting a holistic review of the current repository, paper, experiments, and reviews, and working step by step to build quality code that is more efficient and optimized, cleaner, more concise, more generalizble to different experiemnts, and allows us to conduct more experiments particuralt with sheilding. Curious if it makes more sense to do more https://gymnasium.farama.org/environments/toy_text/ games like taxi or more game theoretic or both. Let's go through this step by step, writing unit tests for every function and end2end tests after. 

Let me know if htere any questions. 

## Notes

I definitly think our current sheilding code in the past repository is a major mess that can be optimized and cleaned up. Look at the old repo with a high level view. if we overfit to recreating it with precision, we might get messy agian and underperform.