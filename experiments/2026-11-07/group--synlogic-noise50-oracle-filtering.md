Instruction received: 2026-11-07T22:04:41-05:00 (first observed record; exact message timestamp unavailable)
Experiment group: [SynLogic oracle filtering](group--synlogic-noise50-oracle-filtering.md)

# SynLogic noise50 oracle-discard baseline

Agent ID: `codex:01a1b910-9053-78a1-b9af-50ac39b15422:/root`
Machine ID: `machine:sha256:b658f544cc6b2f70861219809f19614c038f18e4dad79c97105c05512f65bfeb`
W&B: [Training and evaluation](https://wandb.ai/alliance/Noisy-RLVR/runs/q3-synlogic-noise50-oracle-clean-s42-20261107-baseprompt)

## Goal

Measure performance when an oracle identifies and discards every intentionally corrupted training example, leaving only the clean part of the existing 50%-noise dataset. Start from Base rather than a trained checkpoint.

[Experiment index](../README.md)

## Experimental Setup

Fresh Qwen/Qwen3-4B-Base revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`; discard the originally corrupted rows of the frozen SynLogic-Easy noise50 dataset, with no OLR or relabeling. The original 15,610 rows contain 7,805 corrupted rows and 7,805 unchanged clean rows. Preserve this original corruption mask; do not regenerate or reshuffle it. Exclude another 743 clean rows whose prompt or input-constraint fingerprints collide with held-out SynLogic/KOR/BBH/BBEH evaluation puzzles or the existing overlap audit. Keep all official evaluation files unchanged. The resulting 7,062 training rows retain their original native-verifier labels; signature nonmatches do not prove unrestricted semantic disjointness.

The user selected ten passes rather than a fixed 400-iteration match. With the existing drop-last loader, each pass has 44 complete 160-prompt batches (22 remainder rows dropped, with shuffle seed 42): 440 outer iterations and 2,200 AdamW minibatch updates. Preserve eight responses/prompt, optimizer minibatches of 32 prompts, LR 1e-6 without warmup, weight decay .01, token-mean policy loss, clip .2/.2, dual clip 3, reference KL .001, train temperature 1 and validation temperature .6 with eight responses. Use 24,576 prompt and 4,096 response tokens, truncation=error, and explicit enable_thinking=False. The actual Base ChatML prompts are byte-identical to the earlier Base protocol; maximum prompt length is 24,485 tokens. Two data-parallel/FSDP ranks and rollout tensor parallelism one; no loss implementation change.

Native validation preserves all 270 official questions, with the existing additional 254-question validity-audited metric. The unchanged initial Base evaluation is reused from its verified cache; no fresh initial Base inference. Validate every 20 iterations and save at iteration 1, every 10, the endpoint, and allocation drain. Save/validate every rank’s model, optimizer, scheduler, RNG, sampler and label-history state. Initially retain up to two validated native checkpoints. After the step-320 HF quota recovery, retain only the latest full-state validated native checkpoint and mirror that one checkpoint; preserve the final checkpoint and all pre-evaluation rollout snapshots separately. After training, evaluate the final actor on the already authorized KOR Logic/Cipher full250 protocol (16 responses/question) and derive pass@k, using the existing Base cache only. No additional benchmark inference is authorized by this run.

Use exactly two GPUs on gpu35/gpu36, initially allocation 50137 on gpu35; do not touch the other running jobs. The independent sleeping holder ends 2026-11-09T03:15:52Z (November 8, 10:15:52 PM EST); attach with `srun --jobid=50137 --overlap --pty bash`. Preserve it on workload success/failure/stop, without idle renewal. Bound renewal to three non-overlapping 24-hour leases toward this finite target, with new submission only after the previous allocation ends. Check progress each minute, full resources/logging every 15 minutes, and host working-set memory every two seconds. Preserve user stops; bound blind recovery to three failures in six hours or two without a newer checkpoint. Use an independent CPU HF publisher in the authorized `projects/OLR/runs/qwen3-4b-synlogic-noise50-oracle-clean-s42-20261107` namespace.

This is direct finite experiment supervision in the OLR project, whose shared orchestration is not enabled. Existing stopped experiments and their latches are untouched. Older noisy/full-clean references have different saved iteration budgets and overlap handling, so endpoint comparisons do not isolate the oracle filter alone. Full resolved settings and exact data/source hashes are in W&B.

## Sub-experiments

- [Qwen3-4B-Base oracle-clean subset](qwen3-4b-synlogic-noise50-oracle-clean-baseprompt.md)

## Main Finding (Running)

Training in progress: 378/440 completed outer updates; latest native checkpoint 378. Native validation at step 360: 38.89% mean accuracy on all 270 questions with eight responses each, versus cached Base 2.41%. Two training GPUs on gpu35/gpu36 only; benchmark evaluation examples excluded from training. storage_protective_stop_requires_diagnosis.

## Events

| Event | Datetime | Action taken | Resolved |
|---|---|---|---|
| Oracle-discard experiment authorized | 2026-11-07T22:04:41-05:00 | Preserve original mask, exclude benchmark collisions, stage a fresh Base run for ten passes on two GPUs. | Yes |
| Pre-evaluation rollout preservation verified | 2026-11-08T08:54:58.564977+00:00 | Resumed after checkpoint 70 without dropping completed updates; verified full checkpoint 80 and all 1,280 saved rollouts. Snapshots begin at step 80; earlier rollouts are unavailable. | Yes |
| HF publication failure and recovery | 2026-11-08T08:35:35+00:00 | After external repository cleanup, verified all 39 remote payload files for checkpoint 220; switched to single-thread HTTP uploads after Xet rate limiting and resumed the bounded publisher. | Yes |
| HF quota recurrence and bounded recovery | 2026-11-09T00:04:54Z | Quota rejected step 290. Retired only this run's older HF checkpoint 240, verified full step 320 (19 files, both ranks), and restored single-checkpoint publication; training continued. | Yes |
