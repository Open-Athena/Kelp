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
candidates are this same fallback. `generate_edit` returns `None` when the first
token isn't a position token, the position doesn't map to a valid AST span, or the
replacement doesn't produce valid Python — i.e. these are **position/decoding**
failures (implicating #43 edit-position accuracy and #112.4 bracket constraints).

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
2. **Fix edit validity — the biggest lever.** ~88% of sampled edits are unusable.
   Attack the position/decoding failures directly: edit-position accuracy (#43) and
   the bracket-constraint guard (#112.4). Lifting validity multiplies the effect of
   best-of-N, which currently collapses to the fallback.
3. **Then** consider capacity/data. The model already produces improving edits when
   they are valid; a bigger model or more corpus variance is worth trying *after*
   (1) and (2), and should be measured on the realistic eval so the signal isn't
   masked.

**Bottom line:** Kelp repairs realistic corruptions ~27% (best-of-16) today, and
the main ceiling is that most decoded edits are invalid. The most valuable next
work is a realistic eval default + an edit-validity fix — not scaling the model.
