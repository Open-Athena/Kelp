# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal on-TPU step-rate benchmark (explicit prints so output is captured).

Runs the real sharded training step on whatever devices are present and prints
per-step wall time (with block_until_ready for honest timing). Used to validate
#114 sharding on real hardware and to measure the step rate that sets the
phase-2 reuse factor.
"""

import time

import jax

from kelp.training.distributed import bootstrap_distributed

bootstrap_distributed()
print(f"KELPBENCH devices={jax.device_count()} platform={jax.devices()[0].platform}", flush=True)

from kelp.corpus import TOY_CORPUS  # noqa: E402
from kelp.model.config import EditModelConfig  # noqa: E402
from kelp.model.edit_model import init_edit_params  # noqa: E402
from kelp.training.engine import (  # noqa: E402
    EditTrainingConfig,
    EditTrainingState,
    create_edit_data_iter,
    create_edit_optimizer,
    make_edit_train_step,
)
from kelp.training.sharding import (  # noqa: E402
    data_parallel_size,
    make_data_parallel_mesh,
    replicate,
    shard_batch,
)
from kelp.tree.subtree_bank import SubtreeBank  # noqa: E402
from kelp.tree.tokenizer import EditTokenizer  # noqa: E402

MAX_SEQ_LEN = 256
BATCH = 8
STEPS = 30

tok = EditTokenizer(max_seq_len=MAX_SEQ_LEN)
model = EditModelConfig(
    vocab_size=tok.vocab_size,
    hidden_dim=256,
    intermediate_dim=1024,
    num_layers=4,
    num_heads=4,
    num_kv_heads=4,
    max_seq_len=MAX_SEQ_LEN,
)
cfg = EditTrainingConfig(model=model, max_seq_len=MAX_SEQ_LEN, batch_size=BATCH, wandb_project=None)

mesh = make_data_parallel_mesh()
print(f"KELPBENCH mesh_dp={data_parallel_size(mesh)} batch_per_device={BATCH // data_parallel_size(mesh)}", flush=True)

bank = SubtreeBank.from_corpus(TOY_CORPUS)
data_iter = create_edit_data_iter(corpus=TOY_CORPUS, bank=bank, tokenizer=tok, config=cfg)

opt = create_edit_optimizer(cfg)
params = init_edit_params(model, key=jax.random.PRNGKey(0))
state = replicate(EditTrainingState(step=0, params=params, opt_state=opt.init(params), key=jax.random.PRNGKey(0)), mesh)
train_step = make_edit_train_step(model, opt)

times = []
for step in range(STEPS):
    batch = shard_batch(next(data_iter), mesh)
    t0 = time.perf_counter()
    state, metrics = train_step(state, batch)
    metrics["loss"].block_until_ready()
    dt = time.perf_counter() - t0
    times.append(dt)
    tag = " (compile)" if step == 0 else ""
    print(f"KELPBENCH step={step:02d} dt={dt * 1000:.1f}ms loss={float(metrics['loss']):.3f}{tag}", flush=True)

steady = times[1:]
avg = sum(steady) / len(steady)
tokens = BATCH * MAX_SEQ_LEN
print(
    f"KELPBENCH SUMMARY steady_step={avg * 1000:.1f}ms steps_per_s={1 / avg:.1f} "
    f"examples_per_s={BATCH / avg:.0f} tokens_per_s={tokens / avg:.0f} "
    f"(compile_step={times[0] * 1000:.0f}ms)",
    flush=True,
)
print("KELPBENCH DONE", flush=True)
