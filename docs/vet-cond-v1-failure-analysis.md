# Why `vet-cond-v1` scores ~2% on MBPP — failure analysis

**TL;DR (revised after measurement).** The model *can* repair — the ~2% was
suppressed by two **fixable** causes, not by a fundamental inability:

1. **The eval posed an unrealistically hard task.** Repair quality is *monotonic*
   in corruption difficulty: at the baseline (3 subtree swaps from a 92K unrelated
   bank) best-of-16 is 5.3%, but at **one local edit** it is **27.3%** — ~5× higher.
2. **Decoding emits mostly invalid edits.** ~**88%** of the model's sampled edits
   fail to apply and are discarded (the beam then keeps the program unchanged,
   which *looks* like a no-op). When an edit **is** valid it improves the program
   ~**89%** of the time. So the model has learned useful repairs; the bottleneck is
   edit *validity*, not edit *quality*.

> Honesty note: the first version of this doc read the failures as "no-op
> collapse / the task is unsolvable." A decoding probe overturned that: the model
> never *chooses* a no-op (`no_op = 0` across all samples) — it fails to produce a
> valid edit and falls back to the input. Measurement changed the conclusion.

## Evidence

### 1. Corruption-difficulty sweep (MBPP, step-30000, 50 tasks, best-of-16)

| Corruption | avg pass | best-of-16 | exact | valid |
|---|---|---|---|---|
| steps=3, big bank (baseline) | 2.0% | 5.3% | 0% | 100% |
| steps=2, big bank | 3.1% | 10.0% | 0% | 100% |
| steps=1, big bank | 8.1% | 18.7% | 0% | 100% |
| **steps=1, small (in-context) bank** | **11.1%** | **27.3%** | 0% | 100% |

Repair rises sharply as corruption becomes local and in-context. The baseline was
testing "regenerate the exact original from a heavily-mangled shell", which is much
harder than "fix one realistic edit".

### 2. Decoding diversity (6 corrupted programs × 12 samples @ T=0.8)

| | total |
|---|---|
| valid edits produced | 9 / 72 (**12%**) |
| edits that failed to apply (`None`) | 63 / 72 (**88%**) |
| valid edits that *improved* the program | 8 / 9 (**89%**) |
| samples where the model *chose* a no-op | **0** |

The "no-op" we observe is a **fallback**: when every sampled edit is invalid, the
beam keeps the unchanged program. `best-of-N` is largely wasted because most
candidates are this same fallback.

**Why the edits fail** (instrumented, 144 samples):

| failure cause | share |
|---|---|
| **bad position** (position token → no valid AST span) | **70%** |
| valid edit | 23% |
| invalid replacement (doesn't produce valid Python) | 7% |
| first token not a position token | 0% |

So the dominant failure by far is the **edit position**: the model emits a
position token that does not correspond to a valid AST-node boundary in the
program, so the edit can't be applied. Replacement decoding is rarely the problem
(7%). This points squarely at **edit-position accuracy (#43)**, and suggests a
concrete fix: **constrain the position token at decode time to valid AST
boundaries** (mask invalid positions in the logits), exactly as replacement tokens
are already bracket-constrained. That alone would convert most of the 70% into
applicable edits.

### 3. Training corruption severity (200 corpus programs)

| corruption steps | median % chars changed | ≥50% changed |
|---|---|---|
| 1 | 9.5% | 1.7% |
| 3 (eval baseline) | 21.6% | 4.0% |
| 5 (curriculum max) | 24.0% | 10.0% |

The *typical* corruption is moderate (~10–24% of characters), not total — so
"most of the program is destroyed" is an overstatement. But corruption replaces
whole AST subtrees, so a ~20%-char change can still swap the **load-bearing
expression** for unrelated code, which is semantically fatal (see examples).

## What the corruption looks like (why the hard setting fails)

Corruption replaces subtrees with subtrees from *other* programs. Example
(task 605, "is prime"): `(num % i) == 0` → `'timeStamp' in rec`, and
`return True` → `return cls(api_id=joke['id'], value=joke['value'], ...)`. To
"repair" that at 3 steps the model must invent the exact original from a one-line
docstring — underdetermined. At **1 local edit**, the change is small and the
surrounding intact code constrains the fix, which is why the number jumps to ~27%.

## Implications for the next experiment (revised, and more actionable)

1. **Make the default eval realistic, and report the curve.** Default to ~1 local
   corruption with an in-context/small bank, and report repair-vs-difficulty rather
   than a single hard point. (This is #77; the sweep above is the first cut.)
2. **Fix edit validity — the biggest lever.** ~88% of sampled edits are unusable,
   and **70% of those are bad positions**. Constrain the position token to valid
   AST boundaries during decoding (a decode-time mask, like the existing bracket
   constraint on replacements), and improve edit-position accuracy (#43). Lifting
   validity multiplies the effect of best-of-N, which currently collapses to the
   fallback. (Replacement decoding / #112.4 is only ~7% — lower priority.)
3. **Then** consider capacity/data. The model already produces improving edits when
   they are valid; a bigger model or more corpus variance is worth trying *after*
   (1) and (2), and should be measured on the realistic eval so the signal isn't
   masked.

**Bottom line:** Kelp repairs realistic corruptions ~27% (best-of-16) today, and
the main ceiling is that most decoded edits are invalid. The most valuable next
work is a realistic eval default + an edit-validity fix — not scaling the model.

## Update: the position-mask prototype was measured — neutral (valid ≠ correct)

We prototyped the decode-time position mask (constrain the position token to valid
AST boundaries) and A/B'd it on the cluster (50 tasks each, no retraining):

| setting | unconstrained | constrained |
|---|---|---|
| realistic (steps=1, small bank) | 27.3% | **26.7%** |
| hard (steps=3, big bank) | 5.3% | **8.0%** |

Edit **validity** jumped from 23% → **90%** and "improving" edits 32 → 94, yet
functional repair barely moved (neutral on realistic, +2.7pt on hard). Conclusion:
**valid ≠ correct** — forcing a valid position just relocates the model to its best
*valid* spot, which is often the wrong one, and drops the useful no-op fallback.
The model's position *calibration* is the bottleneck, and inference masking can't
fix it. The mask is kept as an off-by-default research flag.

**This sharpens the plan:** the lever is the **training task**. Highest-value next
step (and the natural continuation of #77): replace alien subtree-swap corruption
with **realistic, in-context near-miss corruption** — apply e-graph *near-miss*
rewrites (e.g. `+`↔`-`, `<`↔`<=`, drop a `not`) to the program's own expressions,
so the model learns to localize and fix plausible single-operator bugs. Retrain,
then re-measure on the realistic eval.
