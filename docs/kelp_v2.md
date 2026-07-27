# Kelp v2: Tree Diffusion for Program Repair, at Scale

*Supersedes [kelp.md](kelp.md) (kept for history). Written after exp10; informed by
the adversarial review tracked in chainlink #141–#145.*

## 1. Why a new design doc

The original design ([kelp.md](kelp.md)) proposed adapting Marin's pretrained 8B
model into a tree-diffusion program synthesizer, trained on a "quine" dataset and
evaluated on MBPP + HumanEval via Marin's executor. That is not the project we
built. Ten experiment cycles later, the actual system is: from-scratch 10M–305M
edit models with a byte+AST-position tokenizer, realistic in-context corruption
over a curated library corpus, streaming synthesis on TPU via Iris, and an
MBPP corruption-repair eval. The v1 doc now misleads more than it guides.

This document restates the goal, re-grounds the design in what the source paper
actually demonstrates, gives an honest account of where the current system
matches and diverges from that recipe, and lays out milestones to get from the
code on `main` today to the stated vision.

**The goal** (from the README): validate **diffusion on syntax trees as a program
repair strategy for real-world Python code** — scaling Kapur et al.'s
tree-diffusion work from 8-primitive inverse-graphics DSLs to a real language —
and measure how repair quality scales with corpus, model size, and compute.
Long-term: a repair-driven **generic coding agent**, where synthesis, editing,
and debugging are all the same operation (denoising toward a specification).

## 2. What the paper actually establishes

Re-reading Kapur, Jenner & Russell (2024), *Diffusion On Syntax Trees For
Program Synthesis* ([arXiv:2405.20519](https://arxiv.org/abs/2405.20519),
[code](https://github.com/revalo/tree-diffusion)) — the method has four
load-bearing components. Any faithful scale-up must have an answer for each.

1. **Small-step forward process.** Corruption = replace a subtree of ≤ σ_small
   primitives with a grammar-sampled subtree of the same nonterminal (σ_small=2).
   Programs are corrupted with **s ~ Uniform[1, 5] sequential mutations**.

2. **Reverse-path training targets.** They explicitly reject "invert the last
   mutation" as noisy (mutations partially cancel). Instead: compute the
   shortest edit path from the corrupted program back to clean (linear-time tree
   diff), pick a **random state along that path** as the input, and supervise
   the **single next path step** as the target. Additionally, with ρ = 0.2 the
   "corrupted" input is a **completely random program** — this teaches
   long-range navigation, i.e. synthesis, not just local repair. Their ablation
   (Fig 5) shows both choices matter.

3. **Observation conditioning — the model sees its own execution.** The denoiser
   is q(z_{t−1} | z_t, **x_t**; **x_0**): every step is conditioned on the goal
   observation x_0 (target image) *and* the current program's execution output
   x_t (its render), plus the difference |x_t − x_0|. The ablation removing x_t
   collapses performance. **This is the paper's central claim** — "our model can
   see the execution result and debug" — not an implementation detail.

4. **Constrained decoding + search.** Position logits are masked to actual tree
   nodes; replacement logits are grammar-masked — validity is guaranteed, not
   learned. Inference is iterative denoising wrapped in search (beam 64,
   ranked by accumulated policy log-prob; the value network exists in the paper
   but is disabled in the released code), with success = observation match and
   the reported metric = problems solved vs. compute (node expansions).

Their stated limitation is our thesis: *"we operate on expressions with no
variable binding, loops, strings, continuous parameters… extending it needs more
work and careful design."* Kelp is that extension.

## 3. The translation table: paper → Python

Scaling to a real language means finding the Python analog of each component.
This table is the core of the v2 design; the third column is the current status.

| Paper (CSG2D/TinySVG) | Kelp v2 (Python) | Status on `main` |
|---|---|---|
| Program z (CFG tree, ≤8 primitives) | Python AST of a real function | ✅ works |
| Small mutation (σ≤2 subtree swap) | Realistic in-context edit (near-miss operator flips, bank swaps bounded by `max_edit_stmts`) | ✅ v9's realistic-or-drop corruption |
| s ~ U[1,5] mutations + reverse-path targets | Multi-step corruption + `find_path` random-step supervision | ⚠️ implemented (`training/generation.py`) but **disabled by exp10** (`max_corruption_steps=1`) — #143 |
| ρ=0.2 random-init mixture (long-range) | `p_random=0.2` random corpus program → clean path | ⚠️ exists but silent, no CLI flag, never ablated — #143 |
| Goal observation x_0 (target image) | **The specification: test asserts / I-O pairs**, plus NL intent (docstring/task text) | ❌ only NL intent, at p=0.5 — #144 |
| Current render x_t (execution feedback) | **Execution result of the current candidate: failing assert, traceback, wrong value** | ❌ absent; model edits blind — #144 |
| \|x_t − x_0\| difference planes | First failing assert + error message vs. expected | ❌ absent |
| Grammar-masked decoding | Byte-level decode + AST-position tokens; position masking to node boundaries | ⚠️ validity is 100% in practice but position constraint is a research toggle — #138 |
| Beam search over edits | `best_of_n` iterative rollouts (max_depth=10) + beam | ✅ structure right; ranked by log-prob only |
| goal_reached (IoU ≥ 0.99) | All task tests pass | ⚠️ used only for **post-hoc** best-of-16 reranking, never during search |
| Metric: % solved vs. node expansions | % tasks fully repaired vs. model calls / candidates | ❌ we report partial-credit best-of-16 pass rate, no CIs — #142 |

Two readings of this table:

- **What's already real:** the representation layer (AST edits, byte+position
  tokenizer, subtree bank, realistic corruption) and the infra layer (streaming
  synthesis, TPU training, resumable eval) are genuine scale-ups of the paper —
  real Python, real libraries, 100% validity maintained from 10M to 305M params.
- **What's missing is the middle of the table.** We kept the paper's *structure*
  (edits on trees) and dropped its *signal* (the model observing goal and
  execution state each step). The v6–v10 walls — 0% exact match, "valid ≠
  correct", matched ≈ unmatched, capacity flat — are exactly what this table
  predicts for a model that cannot see what behavior it is restoring.

## 4. Lessons the ten experiments actually bought us

Compressed from the README history, restated as design inputs:

1. **Validity is solved and cheap.** 100% syntactic validity at every scale.
   This was the paper's guarantee and it transfers to Python. Stop optimizing
   it; stop headline-reporting it.
2. **The task formulation dominates.** v9's move to realistic in-context
   corruption + curated corpus (2% → ~15%) was worth more than any
   model change before or since. Corollary: eval realism is next (§6, M4).
3. **Intent conditioning is necessary but not sufficient.** v6 proved
   unconditioned repair is underdetermined; v7/v8 added docstring prompts;
   correctness stayed low. The conditioning-off ablation has *still* never run.
4. **Capacity is not the current lever** — at this data scale, LR, and
   conditioning. exp10's 305M ≈ 115M (within noise, n=50) tells us the 115M
   model already saturates the training signal (train loss 0.05), not that
   scale never matters. Don't re-run capacity until the signal is richer (M5).
5. **Statistical rigor is not optional.** 50-task evals with triple-selected
   metrics and no CIs produced conclusions the data cannot support (#142).
   Every future milestone defines its metric and n up front.
6. **Eval infra must be adversary-proof.** Executing model-generated code
   in-process with a one-shot SIGALRM lost 17/50 tasks once and remains
   escapable (#141). Anything that runs generated code runs it in a subprocess.

## 5. v2 system design

### 5.1 What we keep unchanged

- **Representation:** byte-level tokens + AST position tokens + prompt prefix;
  single-edit prediction head (position, then replacement bytes). ~1.3K vocab.
- **Forward process:** `corrupt_realistic` (near-miss operator flips, in-context
  swaps, realistic-or-drop) shared verbatim between training and eval.
- **Data engine:** streaming synthesis (seeded, pure, worker-parallel), reuse
  shuffle buffer, curated + Stack Edu corpus sources, decontamination.
- **Infra:** Iris/TPU launch path, GCS Orbax checkpoints, preemption-resumable
  training and (fingerprint-sharded) eval, W&B-optional logging.

### 5.2 What changes

**(a) Restore the diffusion recipe.** Multi-step corruption (s ~ U[1, S], S=3–5
for real code) with reverse-path random-step targets — the machinery already in
`training/generation.py` — becomes the default again. `p_random` becomes a
first-class, documented, ablatable flag. Single-edit (S=1) remains available as
a *control*, not the recipe. Rationale: #143; the paper's Fig 5 ablation; and
the fact that inference is a 10-step iterative loop whose states after step 1
are off-distribution for an S=1 model.

**(b) Spec conditioning (x_0).** The prompt prefix grows from
`[PROMPT] docstring` to a structured spec block:

```
[PROMPT_START] nl_intent [SPEC_START] assert_1 ⏎ assert_2 … [SPEC_END] [PROMPT_END]
```

At training time the spec is synthesized from the *clean* program: execute it on
generated/extracted inputs to record I-O pairs, or lift doctest/docstring
examples. (The clean program is our renderer — we can always produce ground-truth
observations from it, exactly as the paper renders z_0 to get x_0.) At eval time
the spec is the MBPP asserts — which we already possess and today show the model
only after generation, as a rerank filter. Train with spec dropout (p_spec,
p_prompt independently) so the model degrades gracefully without a spec.

**(c) Execution feedback in the loop (x_t and |x_t − x_0|).** After each applied
edit in a rollout, execute the candidate against the spec **in a sandboxed
subprocess** and feed back a compact observation:

```
[OBS_START] status (pass_count/total) ⏎ first_failing_assert ⏎ exception_head [OBS_END]
```

This is the direct analog of the paper's stacked current+target+diff planes, and
it converts blind best-of-16 sampling into guided search. It also gives us the
paper's search-ranking signal for free: rank beam states by tests passing
(their value network's job) with log-prob as tiebreak. Cost control: execution
is per-accepted-edit, not per-token; subprocess pooling amortizes interpreter
startup; a per-test timeout with hard kill bounds the worst case (#141).

**(d) Trustworthy evaluation.** One shared harness (chainlink #99/#100) with:
subprocess-isolated test execution with hard kill-on-timeout; headline metric =
**% tasks fully repaired** (all asserts pass) with bootstrap CIs over tasks,
partial-credit pass-rate as secondary; n=500 tasks for any number that reaches
the README; matched/unmatched arms and corruption-step sweep (1/2/3) reported
together; a real-bug track (§M4) alongside synthetic corruption.

**(e) Then, and only then, scale.** Data first (8.1k → 100k+ spec'd functions),
capacity second (with per-size LR tuning and token budgets that scale with
params), transfer last (Marin models as the capacity vehicle once the recipe is
signal-rich enough to benefit — the v1 doc's transfer idea, in its right place).

### 5.3 Explicitly dropped from v1

- The **quine dataset** and Marin-executor eval integration (superseded by the
  curated + Stack Edu pipeline and the kelp eval harness).
- **Immediate 8B transfer** as the starting point — ten experiments say the
  recipe, not the capacity, is the frontier; transfer is M6, not M1.
- **HumanEval "in a few lines"** — replaced by the spec'd eval plan in M4
  (HumanEvalFix is the natural entry point, not HumanEval synthesis).

## 6. Milestones

Each milestone has an exit criterion and a kill criterion — a result that would
falsify the step so we stop instead of drifting. Issues in parentheses exist.

**M0 — Make the numbers real** *(eval trust; #141, #142)*
Subprocess test runner with hard kill; 500-task protocol with bootstrap CIs;
re-run the exp10 four-cell deconfound (v9 vs exp10 checkpoints × 1-step vs
2-step corruption, timeout fixed, same task set). Re-issue the README v10
conclusions from these numbers.
*Exit:* CI-carrying replacements for every v9/v10 headline claim.
*Kill:* n/a — this milestone is unconditional hygiene.

**M1 — Restore multi-step diffusion** *(#143)*
Default S=3 uniform corruption with reverse-path targets; `--p-random` exposed
and documented; evaluate at corruption-steps 1, 2, 3.
*Exit:* multi-step-trained model ≥ single-edit model at steps=1 (no floor
regression) and strictly better at steps≥2.
*Kill:* if S=1 training matches S=3 training even at 3-step eval, iterative
denoising adds nothing over one-shot repair for Python at this scale — a
publishable negative result that redirects the project toward single-shot
repair models.

**M2 — Spec conditioning (x_0)** *(new; extends #82/#84's prompt work)*
Spec block in the tokenizer; spec synthesis (execute clean program → I-O pairs)
in the data engine; MBPP asserts as eval spec; and the long-owed ablation grid
{no prompt, NL only, spec only, NL+spec} — v8's unrun experiment, finally.
*Exit:* fully-repaired% (M0 metric) improves over NL-only with non-overlapping
CIs; ablation attributes the gain.
*Kill:* if spec conditioning moves nothing, the correctness wall is not an
information gap — capacity/decoding become the prime suspects again and M5
jumps the queue.

**M3 — Execution in the loop (x_t)** *(new; the paper's core mechanism)*
Per-edit sandboxed execution; observation block fed to the next step; beam
ranking by tests-passed. Compare guided rollouts vs. blind best-of-16 at equal
model-call budget (the paper's "solved vs. expansions" curve, ported).
*Exit:* guided search dominates blind sampling at every budget point.
*Kill:* if execution feedback doesn't help a spec-conditioned model, our
observation encoding is wrong (revisit format) — the paper's ablation says this
signal is too valuable for a true null.

**M4 — Real bugs, more data** *(#127, #128 feed this)*
Scale the spec'd corpus toward 100k+ functions (Stack Edu + library sources,
require-corruptible + docstring/spec filters). Add a real-bug eval track —
HumanEvalFix and/or BugsInPy-style single-function bugs — so "repair" stops
meaning only "invert our own corruption operator". Measure the repair-vs-corpus-
size scaling curve (the README's scaling-laws goal, currently unmeasured).
*Exit:* a corpus-scaling curve with ≥3 points, and a real-bug number alongside
the synthetic one.
*Kill:* flat corpus-scaling curve with the M2/M3 recipe → data diversity is not
the binding constraint; stop paying for corpus work.

**M5 — Capacity, done properly** *(reruns exp10's question fairly)*
Only after M2–M4: re-test 115M vs 305M (vs larger) with per-size LR sweeps and
token budgets scaled to params, on the richer signal. exp10 answered "does
capacity help a blind, data-saturated model" (no); M5 asks the real question.
*Exit:* a capacity curve with CIs on the M0 metric.
*Kill:* still flat → from-scratch scaling is exhausted; M6's transfer becomes
the only capacity path.

**M6 — Toward the agent** *(v1's vision, re-entered with evidence)*
Three tracks, gated on M1–M5 outcomes: (i) long-range synthesis — grow
`p_random` / start-from-stub rollouts so the same model synthesizes, not just
repairs (the paper's ρ-mixture, scaled up); (ii) scope growth — multi-function
and file-level edits (seq len, position-token range, and tree-diff over
modules); (iii) capacity via transfer — initialize from a pretrained code model
(Marin), preserving the edit head and constrained decoding. Entry criterion for
"agent" claims: competitive repair on a real-bug benchmark, not corruption
inversion.

Sequencing note: M0 unblocks everything and is pure engineering. M1–M3 are the
science path and each is a 1–2 week TPU-scale experiment with the existing
infra. M4 runs concurrently with M2/M3 (data work is parallel). M5/M6 are
gated, not scheduled.

## 7. Risks and open questions

- **Sequence budget.** 1024 tokens already pinches: spec + observation blocks
  compete with code bytes. Mitigations: observation truncation policy, seq-len
  1536/2048 at M3 (cost is known from the MFU work), or spec summarization.
  File-level scope (M6) likely forces a tokenizer rethink.
- **Execution cost & safety.** M3 multiplies test executions by rollout depth.
  Subprocess pooling and per-test kill-timeouts (#141) are prerequisites, not
  optimizations. All generated code runs sandboxed, always.
- **Spec synthesis quality.** Training specs come from executing clean corpus
  functions; impure/IO-bound functions won't yield clean I-O pairs. The
  require-corruptible filter gains a require-executable sibling; coverage
  fraction is a tracked metric, and spec dropout keeps the model robust to
  spec-free inputs.
- **Synthetic-corruption ceiling.** Even "realistic" corruption is our own
  operator distribution; matched ≈ unmatched hints the model generalizes past
  it, but only the M4 real-bug track can confirm the skill transfers.
- **When to abandon from-scratch.** Standing decision rule: the first time a
  milestone's kill criterion points at capacity twice in a row, we stop
  training from scratch and execute the transfer plan (M6.iii).

## 8. References

- Kapur, Jenner, Russell (2024). *Diffusion On Syntax Trees For Program
  Synthesis.* arXiv:2405.20519. Code: github.com/revalo/tree-diffusion.
- Kelp experiment history: [README.md](../README.md#experiment-history) v1–v10.
- Adversarial review of exp10: chainlink #141 (eval timeout), #142 (statistical
  rigor), #143 (single-edit rationale), #144 (observation conditioning),
  #145 (branch hygiene).
- Original design: [kelp.md](kelp.md) (historical).
