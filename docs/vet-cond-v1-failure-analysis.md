# Why `vet-cond-v1` scores ~2% on MBPP — failure analysis

**TL;DR.** The ~2% isn't mainly a capacity or prompt-conditioning problem. The
corrupt→repair **task as configured is close to unsolvable**: corruption replaces
whole expressions with large, *unrelated* code fragments pulled from other Stack
Edu programs, so "repair" means *regenerating the exact original algorithm* from a
mangled shell plus a one-line description. The model has rationally learned that
editing usually makes things worse, so it **returns the input unchanged** (a
no-op) on almost every held-out program. This holds **in-distribution too**
(≈0% exact recovery on training-corpus programs), which rules out "MBPP is
out-of-distribution" as the primary cause.

## Evidence

Two evals of the step-30000 checkpoint plus a local reproduction of 7 MBPP tasks
(full corrupt→repair, best-of-6):

| Signal | Result |
|---|---|
| MBPP (held out) | 100% valid, **0% exact**, ~2% avg / ~5% best-of-16 pass |
| Held-out corpus (in-distribution) | 100% valid, **~0% exact** recovery (same pattern) |
| Local probe: model returns the corrupted input unchanged | **6 / 7 tasks** |
| Local probe: distinct candidates across the rollouts | **~1.3 of 6** (best-of-N buys almost nothing) |

The two evals agreeing (MBPP and in-distribution both ~0% exact) is the key
result: the model fails to reconstruct originals *even for programs from its own
training distribution*, so the failure is in the **repair objective / corruption
model**, not the eval distribution.

## What the corruption actually does (the crux)

Corruption replaces AST subtrees with subtrees sampled from the bank — i.e. from
*other, unrelated* programs. With 3 mutation steps this injects several alien
fragments, producing a Frankenstein program that shares only scaffolding with the
original. Examples (what the model is asked to "repair"):

### Task 605 — "check if the given integer is a prime number"
```python
# CLEAN (target)
def prime_num(num):
  if num >=1:
   for i in range(2, num//2):
     if (num % i) == 0:
                return False
     else:
                return True
  ...

# CORRUPTED (model input): (num % i) == 0  ->  'timeStamp' in rec
#                          return True     ->  return cls(api_id=joke['id'], value=joke['value'], ...)
def prime_num(num):
  if num >=1:
   for i in range(2, num//2):
     if 'timeStamp' in rec:
                return False
     else:
                return cls(api_id=joke['id'], value=joke['value'], categories=joke['category'])
  ...

# MODEL BEST REPAIR: identical to CORRUPTED (no edit)
```

### Task 57 — "largest number formed from the given digits"
```python
# CLEAN
def find_Max_Num(arr,n):
    arr.sort(reverse = True)
    num = arr[0]
    for i in range(1,n):
        num = num * 10 + arr[i]
    return num

# CORRUPTED: arr.sort(reverse=True) -> type(sim.winners_per_type());
#            range(1,n) -> two_hex(b).rjust(2,'0');  num*10 -> end - start
def find_Max_Num(arr,n):
    type(sim.winners_per_type())
    num = arr[0]
    for i in two_hex(b).rjust(2, '0'):
        num = end - start + arr[i]
    return num

# MODEL BEST REPAIR: identical to CORRUPTED (no edit)
```

To "repair" either of these the model would have to *invent* the correct
expression (`(num % i) == 0`, `arr.sort(reverse=True)`, …) from the docstring
alone. A one-line prompt does not determine the specific algorithm — this is the
**underdetermined-repair** problem from v6, and prompt conditioning is too weak a
signal to overcome it.

## The two failure modes

1. **No-op / identity collapse (dominant, 6/7).** The model predicts an edit that
   leaves the program unchanged, and *all* rollouts collapse to that same output
   (≈1.3 distinct candidates of 6). Best-of-N therefore adds no coverage. The rare
   "passes" happen only when a corruption left some test incidentally passing — not
   because the model repaired anything. This is the same behavior flagged back in
   v3 ("the model rationally prefers no-op over random edits on OOD inputs").
2. **Delete-not-reconstruct (rare).** When it does edit (task 602), it collapses
   the corrupted expression to a valid-but-meaningless token (`str2`) rather than
   reconstructing the intended code — valid Python, wrong program.

## Why this happens

- **The corruption is catastrophic and unrealistic.** Swapping in unrelated
  subtrees does not resemble real bugs; it destroys enough of the program that
  recovery requires regeneration, not repair.
- **The objective is underdetermined.** From a mangled shell + a short docstring,
  the correct program is not identifiable, so the loss-minimizing behavior is to
  do nothing (or minimize edits) — exactly what we observe.
- **Training likely taught the no-op.** Training uses the *same* subtree-swap
  corruption; if training corruptions are equally catastrophic, low training loss
  is achieved partly by learning "usually don't edit," which is what transfers.

## Implications for the next experiment

Ranked by expected information per dollar:

1. **Control corruption difficulty (issue #77).** Re-evaluate with *small, local*
   corruptions (1 step; a small/nearby subtree bank) to measure whether the model
   can repair *realistic* bugs at all. If it can, the ~2% is an eval-difficulty
   artifact; if it still no-ops, it's a training/objective problem. Cheapest,
   highest-information next step.
2. **Audit training corruption severity.** Check the distribution of mutation
   sizes the model trains on. If training corruptions are also catastrophic,
   the model is being taught to no-op — fix the corruption process (smaller/local
   edits, curriculum that stays realistic) before scaling anything.
3. **Decoding diversity.** best-of-N is wasted while rollouts collapse to one
   candidate; without a fix, more samples ≠ more coverage. (Also relevant to the
   eval-speed work — no point batching 16 identical rollouts.)
4. **Only then, capacity / more data.** A bigger model (via bf16 → ~300–500M) or
   more corpus variance is worth trying *after* the corruption/objective is sane —
   otherwise it will likely just learn the same no-op faster.

**Bottom line:** before spending on a larger scaled run, fix what the model is
being asked to do. The most valuable next experiment is a corruption-difficulty
sweep (realistic, local corruptions) evaluated both in-distribution and on MBPP —
that tells us whether Kelp can repair at all, which the current setup does not.
