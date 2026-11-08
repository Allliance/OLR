"""Validate the Qwen3-4B full checkpoint, including its matching OLR history."""

import argparse
import ctypes
import fcntl
import json
import math
import mmap
from pathlib import Path

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_correctness(refinement, step):
    diagnostic = refinement.get("early_correctness")
    if diagnostic is None:
        return {}
    missing = diagnostic["missing_through_step"]
    require(diagnostic["version"] == 1 and 0 <= missing <= step
            and diagnostic["through_step"] == step, "Invalid diagnostic coverage")
    batches = diagnostic["batches"]
    require([b["training_update"] for b in batches] == list(range(missing + 1, step + 1)),
            "Missing internal diagnostic updates")
    for batch in batches:
        require(batch["policy_update"] == batch["training_update"] - 1 and len(batch["records"]) == 160,
                "Diagnostic policy position or group count differs")
        for row in batch["records"]:
            prompt = diagnostic["prompts"][str(row["prompt_id"])]
            require(prompt["task"] == row["task"] and prompt["original_noisy"] == row["original_noisy"]
                    and bool(prompt["original_game_data_str"]), "Diagnostic original puzzle data missing")
            history = refinement["refiner"]["history"][str(row["prompt_id"])]
            visit = row["visits"] - 1
            require(history["answers"][visit] == row["answer_key"]
                    and history["rates"][visit] == row["majority_rate"], "Diagnostic/refiner disagreement")
            require(not row["task"].startswith("val/") and type(row["original_noisy"]) is bool,
                    "Invalid diagnostic training provenance")
            for field in ("raw_answers", "answer_keys", "native_rollout_correct", "verification_errors", "training_rewards"):
                require(len(row[field]) == 8, "Incomplete diagnostic response group")
            require(all(math.isfinite(value) for value in row["training_rewards"]), "Nonfinite diagnostic training rewards")
            require(all(value is None or type(value) is bool for value in row["native_rollout_correct"])
                    and (row["native_majority_correct"] is None or type(row["native_majority_correct"]) is bool),
                    "Invalid native correctness verdict")
        noise_missing = diagnostic.get("noise_metrics_missing_through_step")
        if noise_missing is not None and batch["training_update"] > noise_missing:
            require(batch.get("optimizer_update_completed") is True, "Uncommitted training observation")
            metrics = batch["noise_metrics"]
            noisy = [row for row in batch["records"] if row["original_noisy"]]
            expected = {
                "noisy_group_active": (None, len(noisy), len(noisy)),
                "noisy_rollout_assigned_label_match": (
                    sum(key == row["original_assigned_answer_key"] and error is None for row in noisy
                        for key, error in zip(row["answer_keys"], row["canonical_key_errors"])),
                    sum(error is None for row in noisy for error in row["canonical_key_errors"]), 8 * len(noisy)),
                "noisy_rollout_original_verifier_correctness": (
                    sum(value is True for row in noisy for value in row["native_rollout_correct"]),
                    sum(value is not None for row in noisy for value in row["native_rollout_correct"]), 8 * len(noisy)),
            }
            for name, (numerator, denominator, eligible) in expected.items():
                prefix = "noise/" + name + "/"
                actual = metrics[prefix + "numerator"]
                require(0 <= actual <= denominator and (numerator is None or actual == numerator), "Noise numerator mismatch")
                require(metrics[prefix + "denominator"] == denominator and metrics[prefix + "eligible"] == eligible
                        and metrics[prefix + "failures"] == eligible - denominator, "Noise coverage mismatch")
                require(metrics[prefix + "rate"] == (actual / denominator if denominator else None)
                        and metrics[prefix + "rate_defined"] == int(denominator > 0)
                        and metrics[prefix + "coverage"] == (denominator / eligible if eligible else None),
                        "Undefined or incorrect noise rate")
    return {"early_correctness_missing_through_step": missing,
            "early_correctness_records": len(batches) * 160}


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    folder = args.checkpoint
    lock = (folder.parent / (".validation-" + folder.name + ".lock")).open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    step = int(folder.name.removeprefix("global_step_"))
    require(json.loads((folder / "complete.json").read_text())["step"] == step, "Incomplete checkpoint")
    refinement = json.loads((folder / "olr_state.json").read_text())
    diagnostic_summary = validate_correctness(refinement, step)
    from template import PINNED_TEMPLATE_SHA256
    require(refinement.get("prompt_template_sha256") == PINNED_TEMPLATE_SHA256, "Checkpoint template protocol mismatch")
    require(refinement.get("protocol_id", "").endswith("-baseprompt"), "Checkpoint lacks the corrected protocol identity")
    if (folder / "validated.json").exists():
        previous = json.loads((folder / "validated.json").read_text())
        require(previous.get("step") == step and previous.get("validated") is True
                and previous.get("protocol_id") == refinement["protocol_id"]
                and previous.get("prompt_template_sha256") == PINNED_TEMPLATE_SHA256,
                "Existing validation marker belongs to another prompt protocol")
        return
    require(refinement["step"] == refinement["last_reward_step"] == step, "OLR/model step mismatch")
    from refinement import OnlineLabelRefiner
    refiner = OnlineLabelRefiner(**refinement["refiner"]["config"])
    refiner.load_state_dict(refinement["refiner"])
    visits = sum(len(history["rates"]) for history in refiner.history.values())
    require(visits == step * 160, "OLR history is missing processed prompt groups")

    # The controller has strict VM overcommit. Shared file-backed mappings with
    # write permission removed avoid a writable 50-GB private reservation.
    original = torch.UntypedStorage.from_file
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mprotect.restype = ctypes.c_int

    def readonly_mapping(*args, **kwargs):
        storage = original(*args, **kwargs)
        if storage.nbytes() and libc.mprotect(storage.data_ptr(), storage.nbytes(), mmap.PROT_READ):
            raise OSError(ctypes.get_errno(), "Cannot protect checkpoint mapping")
        return storage

    torch.UntypedStorage.from_file = readonly_mapping
    with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
        model = torch.load(folder / "actor/model_world_size_1_rank_0.pt", map_location="cpu",
                           weights_only=False, mmap=True)
        parameters = sum(t.numel() for t in model.values() if isinstance(t, torch.Tensor))
        require(3_900_000_000 <= parameters <= 4_600_000_000, f"Incomplete Qwen3-4B model: {parameters}")
        require(all(any(f"layers.{i}." in key for key in model) for i in range(36)), "Missing model layers")
        def finite(tensor):
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                return True
            flat = tensor.reshape(-1)
            return all(bool(torch.isfinite(flat[start:start + 1_000_000]).all())
                       for start in range(0, flat.numel(), 1_000_000))
        require(all(finite(tensor) for tensor in model.values()), "Non-finite model tensor")
        del model
        optimizer = torch.load(folder / "actor/optim_world_size_1_rank_0.pt", map_location="cpu",
                               weights_only=False, mmap=True)
        require(optimizer["state"] and optimizer["param_groups"], "Missing optimizer state")
        require(all(math.isfinite(g["lr"]) and g["lr"] > 0 for g in optimizer["param_groups"]), "Invalid LR")
        optimizer_states = len(optimizer["state"])
        require(all(float(v["step"]) == 5 * step for v in optimizer["state"].values()), "Optimizer minibatch counters differ from five per outer update")
        require(all(finite(tensor) for state in optimizer["state"].values() for tensor in state.values()),
                "Non-finite optimizer tensor")
        del optimizer
    extra = torch.load(folder / "actor/extra_state_world_size_1_rank_0.pt", map_location="cpu", weights_only=False)
    require(extra.get("lr_scheduler") is not None and extra.get("rng"), "Missing scheduler or RNG")
    loader = torch.load(folder / "data.pt", map_location="cpu", weights_only=False)
    require(isinstance(loader, dict) and loader, "Missing dataloader state")
    result = {"step": step, "validated": True, "model_parameters": parameters,
              "protocol_id": refinement["protocol_id"], "prompt_template_sha256": PINNED_TEMPLATE_SHA256,
              "optimizer_states": optimizer_states, "olr_prompt_visits": visits,
              "optimizer_steps": 5 * step, "all_model_optimizer_tensors_finite": True,
              "components": ["model", "optimizer", "lr_scheduler", "rng", "dataloader", "olr_history"]}
    result.update(diagnostic_summary)
    temporary = folder / "validated.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(folder / "validated.json")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
