Instruction received: 2026-11-07T22:04:41-05:00 (first observed record; exact message timestamp unavailable)
Experiment group: [SynLogic oracle filtering](group--synlogic-noise50-oracle-filtering.md)

# SynLogic noise50 oracle-discard baseline

Agent ID: `codex:01a1b910-9053-78a1-b9af-50ac39b15422:/root`
Machine ID: `machine:sha256:b658f544cc6b2f70861219809f19614c038f18e4dad79c97105c05512f65bfeb`
W&B: [Training and evaluation](https://wandb.ai/alliance/Noisy-RLVR/runs/q3-synlogic-noise50-oracle-clean-s42-20261107-baseprompt)

## Goal

Measure performance when an oracle identifies and discards intentionally corrupted training examples, starting from fresh Base weights.

## Experimental Setup

Keep the original clean half of the frozen SynLogic-Easy 50%-noise training corpus. Of 7,805 clean rows, exclude another 743 matching held-out prompt/input-constraint fingerprints, leaving 7,062 genuine training rows. Preserve all official evaluation sets. No relabeling or OLR. The user selected ten passes: 44 drop-last batches/pass, 440 outer updates and 2,200 AdamW updates, using exactly two GPUs on gpu35/gpu36.

## Sub-experiments

- [Full settings, checkpoint provenance and final results](qwen3-4b-synlogic-noise50-oracle-clean-baseprompt.md)

## Authorized 400-update extension

On 2026-11-09 the user authorized another 400 outer updates, from 440 to 840, with the same scientific settings and two-GPU budget. Checkpoint 440 remains preserved locally and on HF; its evaluations remain saved locally and on W&B; the latest independently verified remote checkpoint is 470. This extension is in progress. Native accuracy at step 500 is 39.352%, versus 41.667% at 440; no improvement is established. The linked run record contains the preservation and continuation details.

## Main Finding (Completed step440 stage)

Final native SynLogic mean accuracy is **41.667%** on 270 questions with eight responses each, versus cached initial Base **2.407%**. The fixed 254-question validity-audited score is **40.404%**.

The oracle-filtered final model scores **44.025% on KOR Logic** and **20.975% on KOR Cipher** at pass@1, versus 29.450% and 1.425% for the original 50%-noise checkpoint. Evaluation uses 250 questions/benchmark and 16 responses/question; all 8,000 new responses were independently rescored. Full pass@k curves, distinct condition colors and a 40-row results table were remotely verified on W&B. Initial-to-final plots use the explicitly labeled cached 50-question subset; no Base inference was repeated.

Legacy references use different training budgets and lack the same benchmark overlap exclusions. These single-seed endpoints do not isolate the oracle filter alone.

Both native checkpoint ranks and all resume state passed validation; the final 20-file, 49.845 GB HF backup was independently verified. All 19 pre-evaluation rollout snapshots from steps 80–440 passed hashes and lineage checks. Steps 20/40/60 were not captured.

## Continuation from step440 to840

Continuation from preserved step440: 530/840 outer updates; latest validated save 530. Exactly400 additional updates authorized, same data/objective and two-GPU budget. The original step440 findings and KOR evaluation above remain preserved. storage_protective_stop_requires_diagnosis.

## Events

| Event | Datetime | Action taken | Resolved |
|---|---|---|---|
| Oracle experiment authorized | 2026-11-07T22:04:41-05:00 | Fresh Base, original corruption mask, ten filtered-data passes, two GPUs. | Yes |
| Rollout observability added | 2026-11-08T08:54:58Z | Continued after checkpoint 70 with unchanged scientific settings; first snapshot at 80. | Yes |
| Checkpoint-safe lease renewal | 2026-11-09T03:16:12Z | Resume full checkpoint 378 in allocation 50146 without discarding completed updates. | Yes |
| Native endpoint validated | 2026-11-09T07:35:00Z | Validate 440 outer/2,200 AdamW updates and both ranks' full state. | Yes |
| Final HF publication | 2026-11-09T07:51:00Z | Independently verify checkpoint 440, all 20 payload files and both ranks after quota recovery. | Yes |
| Final KOR publication | 2026-11-09T07:53:27Z | Independently verify all 8,000 scores, six actual W&B plots, 40 table rows and model/task tags. | Yes |
