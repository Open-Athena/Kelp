# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preemption-resilient per-item result store shared by the eval CLIs.

Each evaluated item's result is written to a durable per-item JSON shard (local
or ``gs://``) as it completes; on restart, completed shards are loaded and their
items skipped, so a preempted eval resumes instead of restarting from item 0.

The shard directory is fingerprinted by the eval config that determines per-item
results, so re-running the same ``--output`` with different parameters (e.g. a
different ``--n-best-of`` or corpus) does not silently reuse stale shards.
"""

import hashlib
import json

from etils import epath


def eval_fingerprint(parts: dict) -> str:
    """Short stable hash of the config that determines per-item results."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def shard_dir(output_path: str, fingerprint: str) -> epath.Path:
    """Per-item shard directory next to the summary JSON, keyed by fingerprint."""
    out = epath.Path(output_path)
    return out.parent / f"{out.stem}-tasks-{fingerprint}"


def load_completed(shards: epath.Path, id_key: str) -> dict[int, dict]:
    """Load completed per-item results, keyed by their integer id."""
    if not shards.exists():
        return {}
    completed: dict[int, dict] = {}
    for f in shards.iterdir():
        if not f.name.endswith(".json"):
            continue
        try:
            r = json.loads(f.read_text())
            completed[int(r[id_key])] = r
        except (ValueError, KeyError):
            continue  # skip a partially-written/corrupt shard; it will be recomputed
    return completed


def write_result(shards: epath.Path, result: dict, id_key: str) -> None:
    """Persist one item's result. ``shards`` must already exist (mkdir once)."""
    (shards / f"item-{result[id_key]}.json").write_text(json.dumps(result))
