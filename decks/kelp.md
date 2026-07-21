---
title: "Kelp"
subtitle: "Teaching a small model to fix code, one edit at a time"
author: "Open Athena"
date: "July 2026"
title-slide-attributes:
  data-background-color: "#1F1E1B"
---

## What is Kelp? {.oa-key}

Kelp is a small AI model that **repairs buggy Python code**.

You give it a function that's *almost* right — one wrong operator, a swapped
variable — and it proposes an **edit** that fixes it. Then it looks again, and
edits again, until the code works.

- It's **small** (~115–300M parameters) and cheap to train.
- It learns **without a labeled dataset of bugs** — it makes its own.
- The goal: a practical, low-cost **coding agent** for automated repair.

## The problem: program repair

Given code that is broken in a small way, produce the fix.

```python
def area(r):
    return 3.14159 * r + r      # bug: should be r * r
```

Why it's hard:

- The space of possible edits is **enormous** — most changes make things worse.
- "Looks plausible" is not the same as "**actually correct**."
- Traditional tools need lots of labeled bug/fix pairs. Those are scarce.

## The idea: "tree diffusion"

Image models learn by **adding noise to a picture, then removing it**. The same
idea works for code — that's **tree diffusion**:

<div class="oa-box">

**Corrupt → Repair.** Take clean code, inject a realistic bug, and train the
model to predict the edit that *undoes* it.

</div>

- Edits happen on the code's **syntax tree** (its grammatical structure), so the
  result is always valid.
- Repair is **iterative**: predict one edit → apply it → look again → repeat.

<div class="oa-key">
Introduced by <strong>Kapur, Jenner &amp; Russell (2024)</strong>,
<em><a href="https://tree-diffusion.github.io/">Diffusion on Syntax Trees for Program Synthesis</a></em>
— shown on <strong>toy graphics programs</strong> (drawing shapes to match a picture).
<strong>Kelp takes that core idea and scales it to real-world Python.</strong>
</div>

## How we train it (no labeled bugs needed)

The data makes itself — this is the key trick:

<div class="two-col">
<div>

1. Start with **clean, working** code.
2. Inject **one realistic bug** (flip `==`→`!=`, swap a variable).
3. Ask the model: *what single edit gets back to clean?*
4. That edit is the training target.

</div>
<div>

The model predicts two things:

- **WHERE** to edit (a position in the code)
- **WHAT** to write there (the replacement)

A standard small transformer, trained on millions of self-made examples.

</div>
</div>

## The setup

- **Model:** a ~115–300M parameter transformer (tiny by modern standards).
- **Hardware:** Google **TPUs**, orchestrated by our **Iris** cluster tooling.
- **Data:** a **curated corpus** of well-documented functions from high-quality
  Python libraries (stdlib, NumPy, SciPy, Flask, JAX, …) — real code, real
  docstrings.
- **Conditioning:** the model sees the function's **docstring + signature**, so
  it knows what the code is *supposed* to do.

# The journey so far {background-color="#1F1E1B" .oa-dark}

## Breakthrough: fix the *task*, not the model

Early runs stalled at ~2% repair. The problem wasn't the model — it was **what
we trained it on** (unrealistic bugs + noisy code).

Fixing the training task changed everything:

| Metric | Before | After |
|---|---|---|
| Functional repair (best-of-16) | ~2% | **~15%** |
| Syntactically valid edits | ~50% | **100%** |

<div class="oa-key">

The lever was **realistic bugs + clean, curated data** — not more parameters.

</div>

## The wall we hit: "valid ≠ correct"

The model now makes edits that are **always valid Python** — but often **not the
right fix**.

- 100% of edits parse and run.
- Only ~15% actually make the tests pass.

<div class="oa-box-solid">

The frontier is no longer *validity* or *data quality*. It's
**localization + correctness**: picking the *right* edit in the *right* place.

</div>

# This week: faster, and a fair test of "bigger" {background-color="#1F1E1B" .oa-dark}

## Making training efficient

We landed a performance overhaul so experiments are cheaper and observable:

- **bf16 compute** — use the chip's fast native number format.
  <span class="oa-badge copper">~2–4× throughput</span>
- **Fused attention** — a memory-efficient kernel; no more out-of-memory walls.
- **MFU logging** — measure *how much of the chip's peak power we actually use*,
  live during training.

<div class="oa-muted">

Along the way we hardened the pipeline: crash-safe logging, checkpoint-resume on
preemption, and a timeout so a runaway test can't hang an evaluation.

</div>

## Experiment 10: two bets

<div class="two-col">
<div>

**Bet 1 — one repair at a time**

Train on **single-edit** bugs. This matches how the model actually fixes code at
run time (one edit, then re-check), sharpening the signal.

</div>
<div>

**Bet 2 — spend the speedup on size**

Use the efficiency win to train a **~300M** model at *the same cost* as the old
115M, and ask:

*Does more capacity move real repair — or is the bottleneck elsewhere?*

</div>
</div>

<div class="oa-key">
Same data, same budget — we change only <strong>task</strong> and <strong>model size</strong>, so the comparison is clean.
</div>

## Where it stands right now

Both models trained to completion overnight (surviving real hardware preemptions
via auto-resume). Training curves:

![](assets/exp10_progress.png)

<div class="oa-muted">
Left→right: training loss, token accuracy, hardware utilization (MFU). The bigger
300M model uses the hardware better (MFU) but, at a fixed budget, doesn't lead on
token metrics — expected. The real test is repair rate.
</div>

## What we hope to learn

The **evaluation is running now** — repair rate for 300M vs 115M on held-out
tasks. Two possible lessons, both valuable:

1. **300M pulls ahead on repair** → capacity *is* a lever; scale up.
2. **300M ≈ 115M on repair** → the wall is the **repair loop / decoding**, not
   model size → that becomes the next experiment.

<div class="oa-box">

Next idea already in flight: **validation during training** — measure real repair
rate every few thousand steps, so we stop flying blind on the metric that matters.

</div>

# Thanks {background-color="#1F1E1B" .oa-dark}

## In one breath

- Kelp **repairs buggy code** by making small, checkable edits — learned by
  **corrupting clean code and undoing it**.
- A **cheap, small** model on TPUs; the big wins came from the **training task
  and data**, not size.
- Current frontier: turning **valid** edits into **correct** ones.

<div class="oa-muted">
Open Athena · project code, experiment logs, and results in the Kelp repo.
</div>
