# Joint PI05 adapters and task bundle provenance

The hot-switch implementation uses one immutable local Base for both tasks. A
full-parameter task A checkpoint and a different task B checkpoint are not a shared
Base. Local export hashes all model/configuration bytes, checks the adapter's
`base_model_name_or_path` against that content, and records the resulting SHA256.
The runtime repeats this validation before loading. No model files are read by
`activate`, `reset`, or `predict`.

## Current validation status

Hardware-free unit tests cover package integrity, target selection, task-specific
normalization, fake resident warmup, and PI05 sampler control flow with a fake
denoiser. These tests do not constitute a real PI05 forward, GPU training, ACP
training, task rollout, or robot validation.

The inspected workstation has CPU-only PyTorch 2.12.0, stock Transformers 5.9.0,
PEFT 0.19.1, and a broken repository `.venv` referring to `D:\Miniconda3`. The stock
Gemma RMSNorm has no PI05 `cond` interface. No model weight snapshot or task dataset
was found at the configured Linux `/nfs` paths. The physical GPU was already busy;
it was not used. These dependencies are recorded evidence, not a validated lock.
No existing environment was modified.

Preserve the original working environment. Build an independent environment from
its known working custom Transformers `direct_url.json` commit, then add a tested
PEFT version without resolving a generic `transformers-dep` over that branch.
`lerobot[pi,peft]` is not a verified installation recipe for this checkout.
Record the successful exact Python, torch/CUDA, custom Transformers commit, PEFT,
and repository revision after the real smoke test. Do not fabricate a lock before
that gate passes.

## Target selection and ordinary training

The existing PI05 PEFT default targets only the FM expert's Q/V projections and
some projection names that do not exist in this branch. It remains available for
old configurations. The hot-switch configurations explicitly select both:

* `model.paligemma_with_expert.paligemma.language_model...self_attn.(q|v)_proj`
* `model.paligemma_with_expert.gemma_expert...self_attn.(q|v)_proj`
* `model.action_in_proj`, `model.action_out_proj`, `model.time_mlp_in`, and
  `model.time_mlp_out`.

`examples/hotswitch/peft_qv.json` and `peft_extended.json` are **configuration
fragments**, not complete dataset/training configurations. The extended variant
adds Q/K/V/O and gate/up/down MLP projections in both Transformers. Both start at
rank 32, alpha 64, dropout zero, no `modules_to_save`, and no vision LoRA. These are
starting values, not optimized hyperparameters. `joint_target_pattern(vision=True)`
adds SigLIP explicitly. Export the exact result of `audit_targets(policy)` as the
bundle's target list; incomplete language, expert, or projection coverage fails.

Use `lerobot.scripts.lerobot_train` for ordinary adapter training. All trainer
losses, data processing, schedulers, and optimizer selection remain in that entry
point. `PeftConfig` now exposes `lora_alpha` and `lora_dropout`. Existing PEFT defaults
remain unchanged when those fields are absent. All PI05 gradient checkpoint calls
now preserve RNG state so nonzero dropout is safe across recomputation. This does
not prove a particular dropout value improves task quality.

The trainer restores an existing adapter with `is_trainable=True` and avoids
wrapping a `PeftModel` twice. Resume with the trainer's checkpoint config and
`resume=true`, so optimizer and RNG state are restored by the existing trainer.
Do not merge each task into a new Base between SFT and ACP updates.

## Real mechanism gate

After selecting a free authorized device, supply one offline NPZ containing real
unbatched observation tensors and real `action` targets, plus a local Base with
saved processors. From the repository, with `PYTHONPATH=src`:

```powershell
python -m lerobot.async_inference.hotswitch_smoke --help
python -m lerobot.async_inference.hotswitch_smoke --base <local-base> --observations <offline-labeled.npz> --output <new-smoke-directory> --device <available-device> --steps 3 --task-text <task-text>
```

The opt-in test rejects missing CUDA or the wrong PI05 Transformers patch. It
trains A then B on the same Base, checks finite updates in language and expert
LoRA B factors and all four action/time projections, hashes the complete frozen
Base before/after, saves both adapters with processors, restores each independently,
continues training, and checks fixed-seed independent/shared/A→B→A outputs at the
declared tolerances. It writes `smoke_result.json` only after all checks pass.
These temporary adapters are mechanism evidence, not formal task models or ACP
acceptance. The smoke input has no fabricated advantage labels.

To test clean-prefix Training-Time RTC mechanics, add `--rtc-max-delay <d>` with
`0 < d < chunk_size`. This changes only the test configuration, performs normal
prefix-sampling training, and checks hard clamp and fixed-prefix outputs before
injection, after independent restore, after merging an independent comparison
copy, and through shared A→B→A. The result records the actual tested prefix lengths
and `rtc_mechanism_passed`; ordinary smoke leaves this flag null. A formal TRC
launcher must require positive RTC evidence covering its configured training delay.
The smoke also verifies an unadapted Base loss and inference forward before PEFT.
Passing this short mechanism test does not establish trained RTC task quality or
ACP acceptance.

## ACP and Training-Time RTC

Formal ACP plus TRC training uses `lerobot.scripts.lerobot_train_trc`, which still
requires PI05, positive `rtc_training_max_delay`, and `acp.enable=true`. The real
ACP indicator field must exist and contain valid binary labels. Its original
validation and losses are unchanged. Record the label provenance and dropout in
`training_source` and `acp`; do not silently disable ACP to make a run succeed.

The sampler now accepts `prefix_steps` and a normalized `(B,d,D)` `action_prefix`.
It sets clean prefix time to zero and clamps that prefix before and after every
flow integration step. Prefix length must not exceed the checkpoint training
maximum. The runtime task manifest also bounds deployment d. Bootstrap uses zero
prefix once per new task epoch; refill uses that task's own paired raw/execution
queue. These controls are separate from legacy gradient-guided RTC, which remains
on its existing inference path and is explicitly rejected by the first resident
backend. Queue aggregation and latency telemetry do not establish RTC support.

## Export and startup contract

`task_bundle.export_task(adapter_path, destination, metadata, base_path)` copies a
trainer `pretrained_model` directory, including saved pre/postprocessors and
normalization state, then writes a verified `manifest.json`. Required metadata:
`task_id`, untagged `task_text`, `action_schema` (ordered `names`, `units`, positive
`fps`), `camera_mapping`, `lerobot_features`, `target_layers`,
`tokenizer_fingerprint`, `rtc`, `acp`, and nonempty `training_source`. The policy
configuration and Base/adapter content fingerprints are filled by export.

Compute `tokenizer_fingerprint` with the helper in `hotswitch_backend`; it hashes
the actual vocabulary, tokenizer pipeline, special tokens, padding/truncation
configuration, and model length. At startup the saved processors' real tokenizer
is checked against that fingerprint. Only the backend adds the configured ACP
positive tag; tagged input task text is rejected. A prompt that would truncate is
rejected rather than silently losing the condition.

`load_backend(registry)` accepts a trusted local JSON with `tasks` (bundle paths),
`base_path`, `device`, `mode` (`shared` or `dual`), and `offline_observations`
(`task_id` to NPZ path). Relative paths resolve against the registry. Startup loads
and warms every task with bootstrap and nonzero deployment prefix paths, then
clears all generated state before marking ready. Runtime selects only registered
IDs. Shared mode loads one Base plus all adapters; dual mode independently loads
and merges each adapter into its own Base for the full-model comparison. Main
shared weights are never merged. Actual loader call counts, storage identities,
device placement, load seconds, and warmup seconds are exposed for benchmarking.

The initial version requires identical structural policy settings, tokenizer,
action order/units/frequency, camera mapping, target layers, and adapter structure.
Task statistics and task text may differ and are selected together under the
runtime lock. Incompatible combinations fail before serving. No hardware is
constructed by task export, loading, warmup, or smoke tests.
