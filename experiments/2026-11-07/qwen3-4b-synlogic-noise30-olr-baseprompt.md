Instruction received: 2026-11-07, America/New_York (exact message time unavailable)
Experiment group: SynLogic-Easy fixed-label noise with OLR

# Qwen3-4B-Base: SynLogic-Easy, 30% fixed noise and OLR

Agent ID: —
Machine ID: —
W&B: Pending verified training startup.

## Goal

Run the newly requested 30% noise + OLR condition on one GPU. Start from untouched Qwen3-4B-Base; preserve every previously stopped experiment and its STOP files.

[Experiment index](../README.md)

## Experimental Setup

Model revision: `906bfd4b4dc7f14ee4320094d8b41684abff8539`. Reuse the existing frozen Qwen2.5-7B modal verified-wrong targets. No teacher-label generation or benchmark training is authorized.

The original 15,610-row training pool overlaps held-out puzzles. Before assigning noise, remove all 1,493 matching training rows using normalized prompts/questions and task-specific input-constraint fingerprints, including region-name normalization, plus every previously documented overlap. The removed rows comprise 600 Star Placement, 497 Number Wall, 388 Skyscraper, five Space Reasoning and three Operation examples. Keep the official 270-question validation file byte-identical. This leaves 14,117 genuine training examples; recompute the same seeded stratified nested-mask procedure over this pool and assign 4,235 frozen incorrect labels (29.9993%). No model-performance filtering is used. Signature nonmatches do not prove unrestricted semantic disjointness.

This pool differs from the older stopped 30% baseline, so a later comparison is not an isolated OLR causal effect. Keep the original 270 native validation scores and an additional 254-question view excluding ten defective Mathador cases and six unsatisfiable Futoshiki questions. All 270 are now excluded from training by the stated audits.

Ten passes contain 88 complete 160-prompt batches each: 880 outer updates and 4,400 AdamW minibatch steps. Use eight responses per prompt, optimizer minibatches of 32 prompts, LR 1e-6, no LR warmup, weight decay 0.01, token-mean policy loss, clipping 0.2/0.2, dual clip 3, reference KL 0.001, shuffle seed 42, training temperature 1, and validation temperature 0.6 with eight samples. OLR records histories during five warmup passes, then begins selection at update 441 with slope threshold 0.05 and historical-majority agreement; rejected candidates retain their frozen assigned target. No online oracle or wrong-label regeneration.

Use 24,576 prompt tokens, 4,096 response tokens and truncation=error. Explicitly render with enable_thinking=False using Base ChatML template SHA17762d2e5131cb2a950db7dd5700c50095547773d3302b52baebe9ae32257a55. CPU verification proved the actual rendered prompts identical to the historical unmodified Base template on every retained training and validation example. The maximum prompt is 24,485 tokens. Keep gradient checkpointing and the previously measured single-MI355X execution profile.

Validate every 20 updates and save at update 1, every 10, the endpoint and allocation drain. Full native checkpoints include optimizer/scheduler, RNG, resumable sampler, and OLR history. Independently validate finite tensors and the exact five Adam steps per outer update. Retain two newest validated saves, the preselection update-440 comparison and final update 880; older intermediate saves are disposable under the user's retention instruction. Original stopped checkpoints are never pruned by this controller.

Use one physical GPU on gpu35/gpu36 only. Reuse one card from existing holder 50130 on gpu35 initially; its other cards belong to the existing retained allocation and are not exposed to this run. Later leases request one GPU, last at most 24 hours and cannot overlap the old lease; cap automatic leases at eight toward this finite target. Preserve the sleeping allocation holder after workload completion or stop, with no idle renewal. Independent supervision checks progress each minute, a host-memory guard checks every two seconds, and bounded recovery preserves run identity. Stop blind recovery at three failures in six hours or two without a newer checkpoint, then diagnose. A user stop blocks training, retries and renewals.

This is direct finite single-experiment training supervision, as in the earlier OLR recipe. No persistent shared experiment queue or multi-agent dispatcher is enabled by this request. Record runtime state outside Git. Publish only this run's complete checkpoints to the existing authorized HF namespace `projects/OLR/runs/qwen3-4b-synlogic-noise30-olr-s42-20261107`, through an independent CPU publisher.

## Main Finding (Running)

CPU preparation and real dataset/configuration checks passed. GPU training has not started; no accuracy or verified-startup result is claimed.

## Events

| Event | Datetime | Action taken | Resolved |
|---|---|---|---|
| New run authorized | 2026-11-07, America/New_York | Prepare the requested one-GPU 30% noise + OLR run; remove benchmark overlaps, retain existing fixed labels and stopped runs. | Yes |
