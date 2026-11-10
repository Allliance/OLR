Instruction received: 2026-11-07T22:04:41-05:00 (first observed record; exact message timestamp unavailable)
Experiment group: [SynLogic oracle filtering](group--synlogic-noise50-oracle-filtering.md)

# Qwen3-4B-Base SynLogic noise50 oracle-clean subset

Agent ID: `codex:01a1b910-9053-78a1-b9af-50ac39b15422:/root`
Machine ID: `machine:sha256:b658f544cc6b2f70861219809f19614c038f18e4dad79c97105c05512f65bfeb`
W&B: [Training and evaluation](https://wandb.ai/alliance/Noisy-RLVR/runs/q3-synlogic-noise50-oracle-clean-s42-20261107-baseprompt)

## Goal

Measure the outcome of an oracle that discards the originally corrupted examples from the existing 50%-noise dataset.

## Experimental Setup

Start from untouched `Qwen/Qwen3-4B-Base`, revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`. Preserve the frozen corruption mask: the original 15,610 training rows contain 7,805 clean and 7,805 intentionally corrupted rows. Discard the latter without regenerating teacher answers. Exclude 743 further clean rows matching known held-out SynLogic/KOR/BBH/BBEH prompt or input-constraint fingerprints; train only on the remaining **7,062 rows**, with their original native-verifier targets. Keep official evaluation sets unchanged. Fingerprint nonmatches do not establish unrestricted semantic disjointness.

Frozen data SHA-256:

- Parent noise50: `775e75fbcb756275292857e07a776a3903fc0a503212807246d7f3e5430f7dd9`.
- Filtered training: `3caf7427ade694da9588f8b364861f9166fba26e4fe0fdaac5c1495d05d266f6`.
- Official validation: `2f7ed91ea34a49e4f56d0c08f97d805d29852fe03c6577dec8e533a7b9c7bfe5`.

The user chose **ten passes**. Drop-last sampling gives 44 full 160-prompt batches/pass, with 22 remainder rows dropped and seeded reshuffling: **440 outer updates / 2,200 AdamW minibatch updates**. Use eight responses/prompt, optimizer minibatches of 32 prompts, LR1e-6 constant, weight decay .01, token-mean policy loss, clipping .2/.2, dual clip3, grad clip1, reference KL .001; GRPO, OLR disabled. Seed42; training temperature1, top-p1, top-k−1. Preserve prompts up to24,576 and responses up to4,096 tokens, truncation=error, thinking=False. Base ChatML is byte-identical to the earlier comparison; template SHA-256 `17762d2e5131cb2a950db7dd5700c50095547773d3302b52baebe9ae32257a55`.

Use two DP/FSDP ranks and rollout TP1 on exactly two GPUs, restricted to gpu35/gpu36. Actual reference parameters retain the verified legacy CPU offload. Validate every20 updates at temperature.6 with eight responses on the unchanged270-question set; also report the pre-existing fixed254-question validity-audited metric. Reuse cached initial Base evaluations. Save at1, every10, final and lease drain; during the original ten-pass stage retain only the latest validated full native checkpoint after verified HF quota recovery. The extension below additionally protects step440. Preserve rollout snapshots separately.

Executed source changed after update70 only to add read-only diagnostics/rollout preservation, with scientific settings unchanged:

- Initial code: [OLR154ebf8](https://github.com/Allliance/OLR/tree/154ebf88dd1313c3fa75d56610e927eccf29c7f7) and [framework731bd3f](https://github.com/Allliance/model-task-align-rl/tree/731bd3f22d045b0c61bc552fa92a58b88d7a51f6).
- Instrumented code: [OLR8146a9d](https://github.com/Allliance/OLR/tree/8146a9d82c311e5c00611f30dae464710e19fa2b) and [framework97e82a5](https://github.com/Allliance/model-task-align-rl/tree/97e82a5292483caec77322bcc41a42909b22016d).
- Independent CPU checks covered instrumentation isolation and replay, full-rank checkpoints, actual remote logging and publication. Claude reviewed the paired actual code diff read-only and found no bug invalidating this completed trajectory.

## Authorized 400-update extension

On 2026-11-09 the user requested preserving checkpoint 440 and continuing for exactly 400 more outer updates, to **840 outer / 4,200 AdamW updates**. Resume the same complete model, optimizer, scheduler, RNG, dataloader and sampler state, with identical data, objective, LR and W&B identity. The 20-epoch loop ceiling accommodates the 840-update stopping target; it does not authorize 20 full passes.

The 20-file checkpoint 440 is pinned in a separate local preservation directory with its original provenance and evaluations. Automatic native pruning is disabled; the supervisor retains 440 plus the newest fully validated recovery state and keeps compressed rollout snapshots separately. HF mirrors preserve 440 alongside the newest published state. A quota failure initially blocked new uploads; after an observed 197.94 GB drop in actual HF storage accounting, one bounded diagnostic published 470. Independent remote verification checked all 39 files across 440+470 and proved all 20 original checkpoint-440 hashes unchanged, at repository revision 788a8557a962314e8da2abbf7450bf1a0be3955c. Publication continues at milestones 600 and 840, with 440 protected; intermediate native recovery saves continue.

This extension is **in progress**. Native evaluations at 460, 480 and 500 are 41.296%, 39.028% and 39.352%, respectively, compared with 41.667% at 440; these early scores do not show improvement. The actual 460 snapshot passed a full 1,280-rollout tensor/text/token/mask audit and correctly records resume 440. The existing KOR Logic/Cipher 250-question protocol is prepared for the final 840 actor, with a separate color and publication namespace preserving the 440 results. No additional benchmark or Base inference was started.

A later shared-storage pause discarded completed updates 541–542 and part of 543 from execution attempt 8; training resumed the complete 540 state. Abandoned scalar logs remain preserved. Removing 305.79GiB of verified unused pretrained-weight cache restored headroom, and the resumed trajectory produced a finite two-rank 550 checkpoint with 2,750 AdamW updates. Scientific settings and the 840-update endpoint remain fixed.\n\n## Main Finding (Completed step440 stage)

Native final accuracy is **900/2,160 =41.667%**, versus cached Base2.407%. The validity-audited score is **821/2,032 =40.404%**. Both ranks' model/optimizer tensors are finite; all2,200 AdamW updates, scheduler, RNG, dataloader, sampler and label-history state are usable.

KOR evaluation preserves the earlier full250 protocol: BF16, TP1, context8192, temperature1, top-p1, top-k−1, 16 responses/question, max2,048 response tokens and per-question seed42+sourceID. Evaluate only the final actor. A separate CPU exporter checked exact saved-BF16 slices against **both ranks (4,411,424,256 state elements)** and passed finite CPU inference; the historical one-rank recipe export helper was not used. Exporter source SHA-256: `baa58a34660294c732831ea223166c63583abcf7daf59be804a071af9e855dc3`.

| Final condition | Step | Logic pass@1 | Logic pass@16 | Cipher pass@1 | Cipher pass@16 |
|---|---:|---:|---:|---:|---:|
| full-clean | 310 | 43.025% | 66.4% | 16.775% | 39.2% |
| noise50 | 400 | 29.450% | 60.0% | 1.425% | 12.8% |
| noise50-olr | 430 | 41.075% | 63.2% | 12.250% | 35.6% |
| noise50-oracle-clean | 440 | 44.025% | 65.2% | 20.975% | 39.2% |

Pass@k for k=1,2,4,8,16 uses the standard combination estimate from16 saved responses. All8,000 oracle responses were independently rescored with the frozen official scorer. The actual six W&B plot files and40-row table were downloaded and checked for values, condition colors and sample counts. Full250 plots compare final checkpoints; separate Base-to-final lines use the same cached50-question subset. No new Base inference.

Legacy references use different training budgets and lack the same benchmark overlap exclusions. These single-seed endpoints do not isolate the oracle filter alone. Oracle Logic pass@16 is slightly below the old clean endpoint (65.2% vs66.4%); Cipher pass@16 matches it (39.2%). Do not describe every metric as improved.

All19 compressed rollout snapshots at80,100,…,440 passed file hashes and execution-lineage checks; full1,280-rollout tensor/text/token/mask audits passed80,400,440. They preserve frozen labels, clean references, corruption flags, actual rewards/advantages and policy/update/config/source provenance. With no corrupted groups, noisy rates have denominator0 and remain undefined. Snapshots20/40/60 are unavailable. No unexpected non-drain rollback occurred; resuming checkpoint378 on the new lease lost no completed updates.

Original native checkpoint440 was independently verified at completion on [HF](https://huggingface.co/datasets/Alliiance/research-checkpoints), namespace `projects/OLR/runs/qwen3-4b-synlogic-noise50-oracle-clean-s42-20261107`, revision `54531684bf1a25c4258ac93acf064544424e9e17`: **20 files /49,845,137,044 bytes**, including both ranks and the final comparison marker. At the end of the original440 stage, training, publication and final-evaluation workers had exited successfully. The continuation reuses current allocation50146 on gpu35, expiring2026-11-10T03:16:12Z; attach with `srun --jobid=50146 --overlap --pty bash`. No idle renewal is enabled; bounded renewals toward the newly authorized840 endpoint preserve the two-GPU cap.

## Continuation from step440 to840

Continuation from preserved step440: 551/840 outer updates; latest validated save 550. Exactly400 additional updates authorized, same data/objective and two-GPU budget. The original step440 findings and KOR evaluation above remain preserved. training_active.

## Events

| Event | Datetime | Action taken | Resolved |
|---|---|---|---|
| Authorized target | 2026-11-07T22:04:41-05:00 | Keep only originally clean rows, exclude held-out collisions, run ten passes on two GPUs. | Yes |
| Observability handover | 2026-11-08T08:54:58Z | Preserve trajectory after70; snapshots begin80; earlier rollout gap documented. | Yes |
| Lease renewal and storage recovery | 2026-11-09T03:16:12Z | Resume validated378 after old allocation expired; keep two-GPU cap and normal storage threshold. | Yes |
| Native training complete | 2026-11-09T07:35:00Z | Verify final440 and full two-rank state. | Yes |
| Final HF backup verified | 2026-11-09T07:51:00Z | Recover quota, coalesce intermediate uploads, independently verify final20-file payload. | Yes |
| KOR plots remotely verified | 2026-11-09T07:53:27Z | Complete500 questions×16 responses, independent scoring, pass@k and colored W&B publication. | Yes |
