"""Export a validated one-rank actor for independent held-out evaluation."""

import gc
import hashlib
import json
import os
from pathlib import Path


def ordinary_state_dict(state):
    """Unwrap complete one-rank/replicated FSDP values without collectives."""
    import torch
    from torch.distributed.tensor import DTensor
    from torch.distributed._shard.sharded_tensor import ShardedTensor

    normalized = state.copy()
    for key, value in state.items():
        if isinstance(value, DTensor):
            if value.device_mesh.size() != 1 and not all(p.is_replicate() for p in value.placements):
                raise ValueError("Actor export cannot reconstruct multi-rank sharded or partial values")
            local = value._local_tensor
            if tuple(local.shape) != tuple(value.shape):
                raise ValueError("Actor export is missing a DTensor shard")
            value = local
        elif isinstance(value, ShardedTensor):
            shards = value.local_shards()
            if (len(shards) != 1 or tuple(shards[0].tensor.shape) != tuple(value.size())
                    or any(shards[0].metadata.shard_offsets)):
                raise ValueError("Actor export requires every element of each model tensor")
            value = shards[0].tensor
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Unexpected actor state entry: {key}")
        normalized[key] = value
    if hasattr(state, "_metadata"):
        normalized._metadata = state._metadata
    return normalized


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def export_checkpoint(checkpoint, destination, base_model):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    checkpoint, destination, base_model = map(Path, (checkpoint, destination, base_model))
    validation = json.loads((checkpoint / "validated.json").read_text())
    step = int(checkpoint.name.removeprefix("global_step_"))
    if validation.get("step") != step or not validation.get("validated"):
        raise ValueError("Actor export requires a complete validated checkpoint")
    source = checkpoint / "actor/model_world_size_1_rank_0.pt"
    digest = file_sha256(source)
    expected = {"step": step, "checkpoint_model_sha256": digest, "dtype": "bfloat16"}
    marker = destination / "export.json"
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError("The existing export belongs to a different actor checkpoint")
        return destination
    if destination.exists():
        raise ValueError("Preserve the incomplete export directory for diagnosis")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(destination.name + f".stage-{os.getpid()}")
    stage.mkdir()
    state = ordinary_state_dict(torch.load(source, map_location="cpu", weights_only=False, mmap=True))
    config = AutoConfig.from_pretrained(base_model)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16,
                                                attn_implementation="eager")
    model.load_state_dict(state, strict=True, assign=True)
    model.to(dtype=torch.bfloat16)
    model.tie_weights()
    # BF16 matches the native rollout worker's weight transfer. This is an
    # evaluation export; the full FP32 optimizer/resume checkpoint is retained.
    model.save_pretrained(stage, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(base_model).save_pretrained(stage)
    del model, state
    gc.collect()
    (stage / "export.json").write_text(json.dumps(expected, indent=2) + "\n")
    stage.rename(destination)
    return destination
