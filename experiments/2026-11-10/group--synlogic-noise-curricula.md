# SynLogic noise timing continuations

Authorized: Qwen3-4B-Base, seed42, two GPUs per case (four total), gpu35/gpu36 only. Each adds400 outer updates, five AdamW updates per outer update. Batch160 prompts x8 responses, minibatch32, LR1e-6, KL0.001; OLR disabled. Preserve each parent's template and frozen corruption assignments.

- Case1: original Oracle440 (not650), then400 updates to840. Additional step s samples round(80 x min(s/40,1)) noisy groups per batch160; hold50% after40 steps.
- Case2: noise50 rerun400, then400 clean-only updates to800. The user corrected the gate to400, without waiting for970. Preserve model, AdamW and scheduler state across one-to-two-GPU recovery.

Both continuations use7062 clean source examples; case1 also uses7055 frozen noisy examples. Known native-validation, KOR, BBH and BBEH input collisions are excluded. Benchmarks remain evaluation-only. No labels are regenerated. Input auditing does not establish semantic uniqueness.

Once complete and validated, rerun400 (q3b4-synlogic-noise50-rerun-s42-20261109) supersedes the Oct31 comparison baseline (synlogic-q3-4b-noise50-s42-20261031-baseprompt). Preserve historical results and their checkpoint identity; never transfer the older baseline's evaluation values to rerun400.

Retain native validation270 x8 every20 steps, actual batch quotas, masked-advantage diagnostics and complete periodic rollout snapshots. Resource paths, source hashes, sampler states and operational controls reside in the dedicated run store.
