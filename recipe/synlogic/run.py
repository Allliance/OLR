"""Resolve and launch one of the seven matched Qwen3-4B SynLogic conditions."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys

CONDITIONS = ("full-clean", "noise90", "noise70", "noise50", "noise30-olr",
              "noise90-olr", "noise70-olr", "noise50-olr", "noise50-oracle-clean", "clean-to-noisy")


def prepared_metadata(root, condition):
    from synlogic.template import PINNED_TEMPLATE_SHA256, PROMPT_PROTOCOL
    path = root / "runs" / condition / "portable-provenance.json"
    metadata = json.loads(path.read_text())
    identity = metadata.get("wandb", {}).get("id")
    if (metadata.get("condition") != condition or metadata.get("prompt_protocol") != PROMPT_PROTOCOL
            or metadata.get("prompt_template_sha256") != PINNED_TEMPLATE_SHA256
            or not metadata.get("protocol_id", "").endswith(f"/qwen3-4b-synlogic-{condition}-baseprompt")
            or not isinstance(identity, str) or not identity.endswith("-baseprompt")
            or f"-{condition}-s42-" not in identity):
        raise ValueError("Prepared provenance must identify this corrected base-prompt condition and a separate W&B run")
    return metadata


def check_checkpoint_protocol(folder, metadata):
    for name in ("olr_state.json", "validated.json"):
        state = json.loads((folder / name).read_text())
        if (state.get("protocol_id") != metadata["protocol_id"]
                or state.get("prompt_template_sha256") != metadata["prompt_template_sha256"]):
            raise ValueError("Refuse to mix a pilot or another prompt protocol into the corrected comparison")


def overrides(root, condition, framework_code):
    # Reuse execution settings that have passed actual MI355X training/recovery;
    # replace the scientific settings with this paper's matched GRPO protocol.
    spec = importlib.util.spec_from_file_location("single_card_config", framework_code / "tools/train_single_card.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = "full-clean" if condition in {"noise50-oracle-clean", "clean-to-noisy"} else condition.removesuffix("-olr")
    values = module.overrides(root, base)
    metadata = prepared_metadata(root, condition)
    gpus = int(metadata.get("budget", {}).get("max_concurrent_training_gpus", 1))
    replacements = {
        "trainer.n_gpus_per_node": str(gpus),
        "actor_rollout_ref.actor.fsdp_config.fsdp_size": str(gpus),
        "data.train_files": str(root / "data/prepared" / (condition + ".parquet")) if condition in {"noise50-oracle-clean", "clean-to-noisy"} else str(root / "data/prepared" / (base + ".parquet")),
        "+data.dataloader_num_workers": "0",
        "data.train_batch_size": "160", "data.gen_batch_size": "160",
        "data.max_prompt_length": "24576", "data.truncation": "error",
        "data.filter_overlong_prompts": "False",
        "data.max_response_length": "4096", "actor_rollout_ref.rollout.n": "8",
        "algorithm.filter_groups.enable": "False",
        "actor_rollout_ref.actor.use_kl_loss": "True", "actor_rollout_ref.actor.kl_loss_coef": "0.001",
        "actor_rollout_ref.actor.clip_ratio_high": "0.2", "actor_rollout_ref.actor.clip_ratio_c": "3.0",
        "actor_rollout_ref.actor.optim.lr_warmup_steps": "0",
        "actor_rollout_ref.actor.optim.weight_decay": "0.01",
        "actor_rollout_ref.actor.ppo_mini_batch_size": "32",
        "actor_rollout_ref.rollout.max_num_seqs": "256",
        "actor_rollout_ref.rollout.max_num_batched_tokens": "28672",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "28672",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": "28672",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": "28672",
        "actor_rollout_ref.rollout.val_kwargs.temperature": "0.6",
        "actor_rollout_ref.rollout.val_kwargs.top_p": "1.0",
        "reward_model.score_type": "acc_without_think_format",
        "trainer.total_training_steps": str(metadata["target_updates"]), "trainer.total_epochs": "10" if condition in {"noise50-oracle-clean", "clean-to-noisy"} else "10000",
        "trainer.experiment_name": metadata["experiment_name"],
        "trainer.default_local_dir": str(root / "runs" / condition / "checkpoints"),
        "hydra.run.dir": str(root / "runs" / condition / "hydra"),
    }
    for i, value in enumerate(values):
        key = value.split("=", 1)[0]
        if key in replacements:
            values[i] = key + "=" + replacements.pop(key)
    values += [key + "=" + value for key, value in replacements.items()]
    values += ["actor_rollout_ref.actor.kl_loss_type=low_var_kl",
               "+data.apply_chat_template_kwargs={enable_thinking:false}",
               "actor_rollout_ref.rollout.do_sample=True",
               "+actor_rollout_ref.rollout.stop_token_ids=[151643,151645]",
               "+olr.enabled=" + str(condition.endswith("-olr")),
               "+olr.start_select_epoch=4", "+olr.slope_threshold=0.05"]
    audit = root / "supervision/split-overlap-audit.json"
    overlap = [item["validation_index"] for item in json.loads(audit.read_text())["records"]]
    if overlap or json.loads(audit.read_text()).get("overlap_count") != 0:
        raise ValueError("Evaluation samples must be excluded from training")
    values.append("+evaluation.overlap_indices=" + json.dumps(overlap, separators=(",", ":")))
    values.append("+evaluation.invalid_indices=[10,11,12,13,14,15,16,17,18,19,170,172,174,177,178,179]")
    values += ["+provenance.protocol_id=" + metadata["protocol_id"],
               "+provenance.prompt_template_sha256=" + metadata["prompt_template_sha256"]]
    if condition == "clean-to-noisy":
        values += ["+curriculum." + key + "=" + json.dumps(value)
                   for key, value in metadata["curriculum"].items()]
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--framework-code", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--calibrate", action="store_true",
                        help="One local worst-length execution probe, without W&B or checkpoints")
    args, extra = parser.parse_known_args()
    root, framework = args.root.resolve(), args.framework_code.resolve()
    suite_root = Path(os.environ.get("SYNLOGIC_SUITE_ROOT", str(root)))
    if (suite_root / "STOP").exists() or (root / "STOP").exists() or (root / "runs" / args.condition / "STOP").exists():
        raise SystemExit("This experiment has a user STOP latch")
    recipe = Path(__file__).resolve().parents[1]
    paths = [str(recipe), str(framework), str(framework / "SynLogic")]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    os.environ["SYNLOGIC_FRAMEWORK_CODE"] = str(framework)
    os.chdir(framework)
    run = root / ("calibration" if args.calibrate else "runs")
    run = run / "runs" / args.condition if args.calibrate else run / args.condition
    run.mkdir(parents=True, exist_ok=True)
    settings = overrides(root, args.condition, framework) + extra
    if args.calibrate:
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer
        from synlogic.template import verify_base_template
        calibration_file = root / "calibration/data/prepared/full-clean.parquet"
        table = pq.read_table(calibration_file, columns=["prompt"])
        expected = json.loads((root / "supervision/prompt-length-audit.json").read_text())["full-clean"]["max"]
        tokenizer = verify_base_template(AutoTokenizer.from_pretrained(root / "model"))
        if table.num_rows != 1 or len(tokenizer.apply_chat_template(
                table.to_pylist()[0]["prompt"], tokenize=True, add_generation_prompt=True)) != expected:
            raise ValueError("The calibration input is not the audited longest complete prompt")
        settings += ["data.train_batch_size=1", "data.gen_batch_size=1",
                     f"data.train_files={calibration_file}",
                     "actor_rollout_ref.actor.ppo_mini_batch_size=1",
                     "actor_rollout_ref.rollout.ignore_eos=True",
                     "+actor_rollout_ref.rollout.min_tokens=4096",
                     "trainer.total_training_steps=1", "trainer.save_freq=0", "trainer.test_freq=0",
                     "trainer.val_before_train=False", "trainer.logger=[console]",
                     "trainer.resume_mode=disable", f"trainer.default_local_dir={run}/checkpoints",
                     f"hydra.run.dir={run}/hydra"]
    tracker = run / "checkpoints/latest_checkpointed_iteration.txt"
    if not args.resolve_only and not args.calibrate and tracker.exists():
        step = int(tracker.read_text())
        if not (tracker.parent / f"global_step_{step}/validated.json").is_file():
            raise ValueError("Resume checkpoint has not passed full-state validation")
        check_checkpoint_protocol(tracker.parent / f"global_step_{step}", prepared_metadata(root, args.condition))
        settings += ["trainer.val_before_train=False"]
    (run / "overrides.json").write_text(json.dumps(settings, indent=2) + "\n")
    sys.argv = [str(Path(__file__).with_name("main.py")), *settings]
    if args.resolve_only:
        sys.argv += ["--cfg", "job", "--resolve"]
    else:
        os.environ["SYNLOGIC_RUN_ROOT"] = str(run)
        for line in Path(os.environ["SYNLOGIC_WANDB_ENV"]).read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith("WANDB_") and "=" in line:
                key, value = line.split("=", 1)
                parts = shlex.split(value)
                os.environ.setdefault(key, parts[0] if parts else "")
        for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR"):
            path = run / key.lower()
            path.mkdir(exist_ok=True)
            os.environ[key] = str(path)
        import torch
        expected_gpus = int(prepared_metadata(root, args.condition).get("budget", {}).get("max_concurrent_training_gpus", 1))
        if torch.cuda.device_count() != expected_gpus:
            raise ValueError("Visible GPU count differs from the prepared experiment budget")
        for index in range(expected_gpus):
            free, total = torch.cuda.mem_get_info(index)
            if total - free > 1024**3:
                raise ValueError("An assigned GPU has an unreconciled existing workload")
        if not args.calibrate:
            metadata = prepared_metadata(root, args.condition)
            identity = metadata["wandb"]["id"]
            if not isinstance(identity, str) or not identity:
                raise ValueError("A genuine logging identity must be prepared for this protocol")
            if os.environ.get("WANDB_RUN_ID", identity) != identity:
                raise ValueError("W&B recovery identity mismatch")
            os.environ.update(WANDB_RUN_ID=identity, WANDB_RESUME="allow", WANDB_INIT_TIMEOUT="90")
    from synlogic.main import main as train
    train()


if __name__ == "__main__":
    main()
