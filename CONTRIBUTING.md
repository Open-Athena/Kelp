# Contributing to Kelp

Thanks for your interest in Kelp — a research project exploring **tree diffusion
for program repair**. Contributions of all kinds are welcome: bug fixes, new
corruption/augmentation strategies, evaluation harnesses, docs, and experiments.

## Development setup

Kelp uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency
management, and targets **Python 3.12**.

```bash
# If you're not a core maintainer, please click the "Fork" button on the repository page
git clone https://github.com/Open-Athena/kelp.git  # substitute for the fork repo
cd kelp
uv sync  --dev         # create .venv and install deps (including the dev group)
source .venv/bin/activate
# optional: set upstream repo (if you're working out of a fork).
# git remote add upstream https://github.com/Open-Athena/kelp.git
```

## Running the tests

```bash
JAX_PLATFORMS=cpu uv run pytest tests/kelp/ -q
```

`JAX_PLATFORMS=cpu` keeps the suite fast and deterministic on machines without a
GPU/TPU. The full suite is ~200 tests and runs on CPU.

## Code health: pre-commit

Code health is managed with [`pre-commit`](https://pre-commit.com/). It runs
[`ruff`](https://docs.astral.sh/ruff/) (lint + format), [`mypy`](https://mypy-lang.org/)
(type checking of `src/kelp`), and a set of file-hygiene checks. The same hooks
run in CI on every push and pull request, so please run them locally first.

Run all hooks against the whole repo:

```bash
uvx pre-commit run --all-files
```

Install the git hook so they run automatically on every commit (recommended):

```bash
uvx pre-commit install
```

`ruff` and `mypy` run through `uv run`, so they use the exact versions pinned in
the dev dependency group — no separate install needed beyond `uv sync`. You can
also invoke them directly:

```bash
uv run ruff check .        # lint
uv run ruff format .       # auto-format
uv run mypy                # type-check src/kelp
```

## Making a change

1. **Branch** off `main` (`git checkout -b my-change`). Please don't push
   directly to `main`.
2. Make your change and **add or update tests**. A good test covers a distinct
   behavioral property rather than restating an existing assertion.
3. Ensure `uvx pre-commit run --all-files` and `pytest` both pass.
4. To sync changes from main, please rebase: `git pull origin main --rebase`. After
   rebasing, we recommend force pushing with lease: `git push --force-with-lease`.
   Please keep tidy commit histories when possible.
5. Open a **pull request** against `main` with a clear description of the *what*
   and *why*. Link any related issue. We merge into main via squash & merge to 
   help keep commit histories tidy.

## Issue tracking

Day-to-day work is tracked with [chainlink](https://github.com/dollspace-gay/chainlink), a lean local
issue tracker (`.chainlink/`), in addition to GitHub Issues. External
contributors should file **GitHub Issues** — that's the canonical place for
public discussion. Use the issue templates where they fit.

If you opt in to using chainlink for your development, let a maintainer know -- it 
might mean it's time to adopt [crosslink](https://github.com/forecast-bio/crosslink).

## Scope & style

- Every intermediate state of a repair must be **valid Python** — corruption and
  edit operations should preserve syntactic validity.
- Match the surrounding code's style, naming, and comment density.
- Keep research experiments reproducible: document data sources and any
  non-default configuration in the PR.

## Code of Conduct

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
