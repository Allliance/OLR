# Qwen3-4B-Base: OLR on SynLogic puzzles

## Current finite protocol: November 7

[Protocol and validated milestones](../../experiments/2026-11-07/qwen3-4b-synlogic-noise30-olr-baseprompt.md). This branch adds a fresh, one-GPU noise30-olr run with a benchmark-disjoint 14,117-row pool, 4,235 fixed wrong labels, explicit thinking-disabled Base ChatML prompts, 880 outer updates and OLR selection from update 441. The original validation set stays intact. The historical experiments above remain stopped; their checkpoints and data are unchanged. This branch does not resume their trajectories.

This run uses one GPU on gpu35/gpu36 and stops at 880 outer updates. The
historical study below remains stopped and grants no new execution or resource
authorization. Its benchmark-overlapping training pool must not be reused.

Current validation keys have explicit, fixed meanings:

- `val-core/synlogic/reward/mean@8`: all 270 official questions, eight responses each.
- `val-core/synlogic-disjoint/reward/mean@8`: the same 270 questions in this run,
  because known training/evaluation overlap is zero. This legacy key was a
  234-question view in the older study; never compare those aliases as if they
  used the same question set.
- `val-core/synlogic-audited/reward/mean@8`: 254 questions after excluding the
  16 independently audited defective items. The official 270 remain unchanged.

Older models used a different, benchmark-overlapping training pool. Matching
evaluation denominators does not make those runs a controlled OLR comparison.

## Historical October 31 study (stopped)

User-authorized on 2026-10-31. Run full-clean and 90%, 70%, and 50% fixed-label
noise without OLR, plus each noisy condition with OLR. Use one GPU per condition
and the eighth for calibration and pass@k diagnostics. This study operates
under direct supervision, outside the parent project's staging shared queue.

## Frozen inputs

Model: `Qwen/Qwen3-4B-Base`, revision
`906bfd4b4dc7f14ee4320094d8b41684abff8539`. Reuse the preceding study's exact
15,610 retained SynLogic-Easy training rows, original targets, frozen synthetic
wrong answers, and nested seed-42 masks: 14,049, 10,927, and 7,805 corrupted
rows. The wrong answers were Qwen2.5-7B's modal verified-wrong answers among
16 frozen rollouts. Do not regenerate them with Qwen3; their activation may
differ for the new policy. Preserve duplicate-row weighting and source IDs.

Each baseline/OLR pair reads the identical prepared file. Clean rows use the
native puzzle verifier; corrupted rows match the normalized extracted answer
to the assigned fixed target. A selected OLR candidate replaces the target for
that rollout group using the same matcher. No reward signs are flipped.
Hidden original labels do not enter the refinement gates.

The older study identified 36 exact-prompt overlaps in the 270-question
validation split and called the remaining 234-question view “disjoint.”
The November 7 audit subsequently identified additional overlaps. That older
label is not a guarantee of semantic disjointness or a valid generalization
comparison with the current run. Preserve the historical metrics for
interpretation; use the current validation definitions above for new analysis.

## Method and schedule

`refinement.py` adapts the released `_get_slope` and historical-majority gate
to canonical puzzle answers. Compute the unique majority among extractable
answers and its rate over all eight responses. Ties and missing answers never
become labels. Compute the least-squares slope over the entire rate history;
select only when it is at least 0.05 and the current answer matches the unique
historical majority. Rejected candidates use the frozen assigned target.

Tie handling is part of the protocol: a non-unique mode or no admissible answer
records a null candidate and pass rate 0.0; that zero remains in the OLS history.
This follows the released weak-noise branch's `__NoMajorityAnswer__` target and
reward-derived pass rate, rather than recording the largest tied count divided
by eight. A zero candidate rate does not mean every rollout is incorrect under
the original verifier; use the separate native-correctness diagnostics.
See the [pinned released implementation](https://github.com/ShenzhiYang2000/OLR/blob/da052ed8807691f6559c1e3ac9f5b4e1244680b1/verl/trainer/ppo/ray_trainer.py#L1557).
The upstream strong-noise branch's online wrong-answer resampling is not used
because this experiment requires fixed labels.

Follow the released ten-epoch GRPO schedule: 160 prompts × 8 responses per
outer iteration, optimizer minibatches of 32 prompt groups, LR 1e-6, no LR
warmup, AdamW weight decay 0.01, token-mean loss, clipping 0.2/0.2, dual clip 3,
and reference KL loss 0.001 (`low_var_kl`). Disable dynamic group filtering.
There are 97 complete batches per pass, hence **970 outer iterations / 4,850
optimizer minibatches** in the first comparison block. Then automatically
resume further matched ten-pass blocks until the user's explicit stop,
preserving the optimizer, dataloader, OLR histories, and W&B identity.
Zero-based `start_select_epoch=4` means five warmup passes, with refinement
from pass six onward. Derive the epoch from completed iterations so recovery
and block continuation do not reset the warmup or history.

Use a 4,096-token response cap and preserve complete prompts with a 24,576-token
allowance and truncation set to error. The measured maximum is 24,485 tokens;
the original long prompts must not be filtered or truncated. The
combined 28,672-token allowance fits the model's 32,768 context. Train with
temperature 1, top-p 1, top-k -1; validate eight sampled responses at temperature
0.6 with the same top-p/top-k. Override the model's shipped greedy generation
config. Stop on 151643 and 151645. Use the unmodified base-model chat template,
as in the released launcher with no thinking kwargs; do not inject an empty
`<think>` block. Use verifier accuracy without a separate reasoning-tag bonus.
Earlier non-thinking pilots have separate logging identities and checkpoints;
the corrected comparison starts from the untouched base weights.

## Execution and recovery

This recipe uses `model-task-align-rl` revision
`ec02977cba8c365ea12725d9e737220e85cc1296` as its calibrated veRL backend.
The OLR adapter lives in this fork; the upstream eight-A100 installation is
not assumed to work unchanged on the current AMD cluster.

```bash
python OLR/recipe/synlogic/run.py \
  --framework-code model-task-align-rl \
  --root /path/to/run-store --condition noise90-olr
```

`--resolve-only` prints the configuration. `--calibrate` is a separate local
one-update probe using the longest prompt and forced full-length responses,
with no W&B run or checkpoint. It does not count toward a condition. Runtime
uses one MI355X, TP/FSDP world size one, gradient checkpointing, BF16 compute
with FP32 optimizer state, eager vLLM generation, resident-engine memory
fraction 0.35, and 256 concurrent sequences. CPU preflight is not GPU validation.

Save at iteration 1, every 10 iterations, the endpoint, and lease drain.
Include model, optimizer, scheduler, RNG, dataloader, and `olr_state.json`.
Validate matching counters and complete refinement histories. Independent CPU
workers publish to the authorized HF mirror under `projects/OLR/runs/`.
Keep two recent checkpoints and the matched 970-iteration comparison milestones
after verifying their remote payloads, and
preserve logging identities on resume. Operational files stay in ignored
scratch or external run storage.

`sampling.py` preserves the original seeded `torch.utils.data.RandomSampler`
order and serializes its iterator and generator state. The framework's legacy
loader saved only a cursor, which silently restarted its shuffle stream on
later-pass resumes. Legacy checkpoints require an explicitly validated
`+data.legacy_sampler_epoch=N`: the local shuffle-generator epoch at that save,
which can differ from the global training epoch after an earlier legacy resume.
Never infer this value from the global step alone. Verify it against the
recorded producer trajectory before migration. New sampler checkpoints are
self-contained and need no such override. Repairing future restores preserves
the observed trajectory; it cannot undo prior repeats or omissions.

The eighth card first probes the longest prompt, then measures frozen-base
pass@k for k=1..256 using 256 responses per validation question. Report both
original and disjoint splits. Derive matched 4k and extended 8k response-cap
curves from the same token streams; distinguish strict answer extraction from
a relaxed semantic check and retain verifier errors as uncertainty bounds.
Matched comparison checkpoints require the same diagnostic comparison.

Maintain minute-level progress/recovery checks and 15-minute resource/logging/
publication audits through the experiment or user stop. Stop blind retries at
three failures in six hours or two without a newer checkpoint; diagnose and
test a correction before resuming. A user STOP disables restart and renewal
while preserving checkpoints and the separate allocation holder. Holders end
in `sleep infinity`; use non-overlapping leases of at most 24 hours and never
reserve more than eight GPUs concurrently. Automatic renewal has an eight-lease
review horizon that the controller cannot reset. Before extending it, the
active agent must record a fresh check of the user's continuing until-stop
authorization, the current goal, STOP latches, and unchanged resource limits.
Continue the requested work until the user's stop. Startup verification is
not study completion.

Sources: [OLR paper](https://arxiv.org/abs/2604.03993),
[released Qwen3 script](../../exp_script/run_qwen3-4B-base_noise_label_weak_use_olr.sh).
