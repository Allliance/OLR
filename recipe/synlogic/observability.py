"""Read-only noisy-training measurements and periodic complete rollout archives."""

from collections import defaultdict, deque
from enum import Enum
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch


def plain(value):
    if isinstance(value, Enum):
        return plain(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(key): ("[REDACTED]" if any(secret in str(key).lower() for secret in
                ("api_key", "access_token", "auth_token", "password", "credential", "secret", "hf_token",
                 "token_file", "token_path", "wandb_env"))
                else plain(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported rollout provenance type: {type(value).__name__}")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_diagnostic(directory, value):
    """Keep superseded attempts before replacing the legacy per-step view.

    One trainer writes this directory. A conflicting record within an attempt
    is an error; a new attempt may replay the step after checkpoint rollback.
    """
    directory = Path(directory)
    step = int(value["training_update"])
    attempt = str(value["attempt"])
    if step < 1 or not attempt.isdecimal():
        raise ValueError("Diagnostic step and attempt must be identified")
    path = directory / f"step_{step}.json"
    if path.exists():
        previous_bytes = path.read_bytes()
        previous = json.loads(previous_bytes)
        if previous == value:
            return path
        previous_attempt = str(previous["attempt"])
        if not previous_attempt.isdecimal() or previous["training_update"] != step:
            raise ValueError("Existing diagnostic identity is invalid")
        if previous_attempt == attempt:
            raise ValueError("Conflicting diagnostic replay within an attempt")
        archived = directory / "superseded" / f"attempt_{previous_attempt}" / path.name
        archived.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, archived)
        except FileExistsError:
            if archived.read_bytes() != previous_bytes:
                raise ValueError("Conflicting superseded diagnostic")
        with archived.open("rb") as stream:
            os.fsync(stream.fileno())
        descriptor = os.open(archived.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    atomic_json(path, value)
    return path


def token_digest(ids):
    return hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio(prefix, numerator, denominator, eligible):
    return {prefix + "/numerator": numerator, prefix + "/denominator": denominator,
            prefix + "/eligible": eligible, prefix + "/failures": eligible - denominator,
            prefix + "/rate": numerator / denominator if denominator else None,
            prefix + "/rate_defined": int(denominator > 0),
            prefix + "/coverage": denominator / eligible if eligible else None}


@torch.no_grad()
def noisy_metrics(advantages, response_mask, uids, records):
    """Active groups use actual nonzero advantages on actual valid response tokens.

    Rollout rates use verified cases as denominator, with eligible totals,
    failures and coverage explicit. Unextractable answers are verified failures
    to match a label; verifier exceptions are unverified, never false accuracy.
    """
    assert advantages.shape == response_mask.shape and len(uids) == len(advantages)
    valid = response_mask.bool()
    if not bool(torch.isfinite(advantages[valid]).all()):
        raise FloatingPointError("Nonfinite masked training advantages")
    active = ((advantages != 0) & valid).any(-1).cpu().tolist()
    indices = defaultdict(list)
    for i, uid in enumerate(uids):
        indices[str(uid)].append(i)
    observations = {row["native_group_uid"]: row for row in records}
    groups = active_groups = rollouts = matches = match_verified = correct = correct_verified = 0
    for uid, positions in indices.items():
        row = observations[uid]
        assert len(row["raw_answers"]) == len(positions), "Generated/trained group mismatch"
        if not row["original_noisy"]:
            continue
        groups += 1
        active_groups += any(active[i] for i in positions)
        rollouts += len(positions)
        for key, error in zip(row["answer_keys"], row["canonical_key_errors"]):
            if error is None:
                match_verified += 1
                matches += key == row["original_assigned_answer_key"]
        for verdict in row["native_rollout_correct"]:
            if verdict is not None:
                correct_verified += 1
                correct += verdict is True
    metrics = ratio("noise/noisy_group_active", active_groups, groups, groups)
    metrics.update(ratio("noise/noisy_rollout_assigned_label_match", matches, match_verified, rollouts))
    metrics.update(ratio("noise/noisy_rollout_original_verifier_correctness", correct, correct_verified, rollouts))
    return metrics


def snapshot_due(step, frequency, total_steps):
    next_step = step + 1
    return step >= 1 and frequency > 0 and next_step <= total_steps and (
        next_step % frequency == 0 or next_step == total_steps)


@torch.no_grad()
def snapshot_payload(batch, records, prompt_registry, tokenizer, metadata):
    required = {"prompts", "responses", "attention_mask", "response_mask", "advantages",
                "token_level_rewards", "token_level_scores"}
    assert required <= set(batch.batch.keys()), "Full rollout tensors are unavailable"
    tensors = {key: value.detach().cpu() for key, value in batch.batch.items()}
    uids = [str(uid) for uid in batch.non_tensor_batch["uid"]]
    observations = {row["native_group_uid"]: row for row in records}
    groups, rollouts, queues = {}, [], {}
    prompt_width = tensors["prompts"].shape[-1]
    for uid, row in observations.items():
        assert len(row["response_token_sha256"]) == len(row["raw_answers"])
        lookup = defaultdict(deque)
        for index, fingerprint in enumerate(row["response_token_sha256"]):
            lookup[fingerprint].append(index)
        queues[uid] = lookup
    for i, uid in enumerate(uids):
        row = observations[uid]
        source = prompt_registry[str(row["prompt_id"])]
        group_id = f'{metadata["protocol_id"]}/update-{metadata["training_update"]}/question-{row["prompt_id"]}'
        if uid not in groups:
            prompt_mask = tensors["attention_mask"][i, :prompt_width].bool()
            groups[uid] = {**plain(source), "group_id": group_id, "question_id": row["prompt_id"],
                "native_group_uid": uid,
                "prompt_text": tokenizer.decode(tensors["prompts"][i][prompt_mask], skip_special_tokens=False),
                "frozen_assigned_label": plain(batch.non_tensor_batch["reward_model"][i]["ground_truth"]),
                "effective_target_mode": "olr_selected_majority" if row["selected"] else ("frozen_corrupted_label" if row["original_noisy"] else "original_verifier"),
                "reward_manager_target_mode": "fixed_wrong_answer" if row["selected"] or row["original_noisy"] else "original_verifier",
                "effective_assigned_answer_key": row["answer_key"] if row["selected"] else row["original_assigned_answer_key"],
                "olr_selected": row["selected"], "olr_answer_key": row["answer_key"],
                "olr_slope": row["slope"], "olr_majority_rate": row["majority_rate"], "visit": row["visits"]}
        mask = tensors["response_mask"][i].bool()
        response_ids = tensors["responses"][i][mask]
        fingerprint = token_digest(response_ids)
        assert queues[uid][fingerprint], "Rollout identity changed after balancing"
        original_index = queues[uid][fingerprint].popleft()
        rollouts.append({"tensor_row": i, "question_id": row["prompt_id"], "group_id": group_id,
            "rollout_id": f"{group_id}/rollout-{original_index}", "native_group_uid": uid,
            "generation_index": original_index, "response_token_sha256": fingerprint,
            "response_text": tokenizer.decode(response_ids, skip_special_tokens=False),
            "extracted_answer": row["raw_answers"][original_index],
            "answer_key": row["answer_keys"][original_index],
            "canonical_key_error": row["canonical_key_errors"][original_index],
            "native_original_verifier_correct": row["native_rollout_correct"][original_index],
            "original_verifier_error": row["verification_errors"][original_index],
            "training_verifier_status_available": row.get("training_verifier_status_available", False),
            "training_verifier_timed_out": row.get("training_verifier_timed_out", [None]*len(row["raw_answers"]))[original_index],
            "training_verifier_unverified": row.get("training_verifier_unverified", [None]*len(row["raw_answers"]))[original_index]})
    assert len({row["rollout_id"] for row in rollouts}) == len(uids), "Rollout IDs collide"
    return {"schema_version": 1, "kind": "complete_training_rollouts", "metadata": plain(metadata),
            "tensors": tensors, "uids": uids, "groups": groups, "rollouts": rollouts,
            "diagnostic_records": plain(records),
            "noise_metrics": noisy_metrics(tensors["advantages"], tensors["response_mask"], uids, records)}


def write_snapshot(root, step, payload):
    """Publish immutable compressed data by atomic directory rename, never overwrite."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"global_step_{step}"
    staging = Path(tempfile.mkdtemp(prefix=f".step-{step}-", dir=root))
    archive = staging / "rollouts.pt.gz"
    with archive.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=1, mtime=0) as compressed:
            torch.save(payload, compressed)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {"schema_version": 1, "kind": "complete_training_rollouts", "training_update": step,
                "policy_update": payload["metadata"]["policy_update"], "scheduled_evaluation_update": step + 1,
                "rollout_complete": True, "groups": len(payload["groups"]),
                "rollouts": len(payload["rollouts"]), "sha256": file_digest(archive),
                "bytes": archive.stat().st_size,
                "tensor_shapes": {key: list(value.shape) for key, value in payload["tensors"].items()}}
    manifest["tensor_dtypes"] = {key: str(value.dtype) for key, value in payload["tensors"].items()}
    atomic_json(staging / "complete.json", manifest)
    if final.exists():
        previous = json.loads((final / "complete.json").read_text())
        if previous["sha256"] != manifest["sha256"]:
            raise FileExistsError(f"A different snapshot already exists; new data preserved at {staging}")
        shutil.rmtree(staging)
    else:
        staging.rename(final)
    return final


def commit_snapshot(path, step):
    path = Path(path)
    manifest = json.loads((path / "complete.json").read_text())
    assert manifest["training_update"] == step
    atomic_json(path / "committed.json", {"training_update": step, "optimizer_update_completed": True,
                                          "snapshot_sha256": manifest["sha256"]})
    tracker = path.parent / "latest_committed_snapshot.txt"
    previous = int(tracker.read_text()) if tracker.exists() else 0
    assert step >= previous, "Snapshot tracker must not roll backward within an attempt"
    temporary = tracker.with_suffix(".tmp")
    temporary.write_text(str(step) + "\n")
    temporary.replace(tracker)


def validate_snapshot(path):
    path = Path(path)
    manifest = json.loads((path / "complete.json").read_text())
    commit = json.loads((path / "committed.json").read_text())
    archive = path / "rollouts.pt.gz"
    assert manifest["sha256"] == commit["snapshot_sha256"] == file_digest(archive)
    assert archive.stat().st_size == manifest["bytes"] and commit["optimizer_update_completed"] is True
    with gzip.open(archive, "rb") as stream:
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    assert payload["schema_version"] == 1 and payload["kind"] == "complete_training_rollouts"
    assert payload["metadata"]["training_update"] == manifest["training_update"] == commit["training_update"]
    assert payload["metadata"]["policy_update"] == manifest["training_update"] - 1
    count = manifest["rollouts"]
    assert len(payload["rollouts"]) == len(payload["uids"]) == count
    assert len(payload["groups"]) == manifest["groups"]
    assert len({row["rollout_id"] for row in payload["rollouts"]}) == count
    for key, shape in manifest["tensor_shapes"].items():
        assert list(payload["tensors"][key].shape) == shape and shape[0] == count
        assert str(payload["tensors"][key].dtype) == manifest["tensor_dtypes"][key]
    metrics = noisy_metrics(payload["tensors"]["advantages"], payload["tensors"]["response_mask"],
                            payload["uids"], payload["diagnostic_records"])
    assert metrics == payload["noise_metrics"]
    for row in payload["rollouts"]:
        i = row["tensor_row"]
        ids = payload["tensors"]["responses"][i][payload["tensors"]["response_mask"][i].bool()]
        assert token_digest(ids) == row["response_token_sha256"]
        assert row["native_group_uid"] in payload["groups"]
    result = {"validated": True, "kind": manifest["kind"], "step": manifest["training_update"],
              "groups": manifest["groups"], "rollouts": count, "sha256": manifest["sha256"]}
    atomic_json(path / "validated.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    print(json.dumps(validate_snapshot(parser.parse_args().snapshot)))
