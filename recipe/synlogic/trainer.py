"""Puzzle-domain OLR integration with complete refinement-state recovery."""

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from src.dapo_ray_trainer import RayDAPOTrainer
from src.fixed_labels import answer_key, extract_answer
from src.syn_logic_reward import Data, SynLogicRewardManager, TaskDeadlineExceeded, verifier_classes, verifier_deadline

from .refinement import OnlineLabelRefiner
from .diagnostics import diagnose


class ValidationRecorder:
    def __init__(self, reward):
        self.reward = reward
        self.records = []

    def __call__(self, data, return_dict=False):
        result = self.reward(data, return_dict=True)
        scores = result["reward_tensor"].sum(-1).tolist()
        for i, score in enumerate(scores):
            self.records.append((int(data[i].non_tensor_batch["extra_info"]["index"]), score))
        return result if return_dict else result["reward_tensor"]


class RefinementReward:
    def __init__(self, trainer, reward):
        self.trainer = trainer
        self.reward = reward
        self.refiner = OnlineLabelRefiner(**dict(trainer.config.olr))
        self.last_step = 0
        self.metrics = {}
        self.cached_batch = None
        self.cached_reward = None
        self.correctness = {"version": 1, "missing_through_step": 0, "through_step": 0,
                            "prompts": {}, "batches": []}
        self.diagnostic_sha256 = hashlib.sha256(Path(__file__).read_bytes()
            + Path(diagnose.__code__.co_filename).read_bytes()).hexdigest()

    def __call__(self, data, return_dict=False):
        step = self.trainer.global_steps
        # The native loop may retry a reward call to accommodate old managers.
        # Never append the same batch's trajectory twice.
        if data is self.cached_batch:
            return self.cached_reward if return_dict else self.cached_reward["reward_tensor"]
        if step <= self.last_step:
            raise RuntimeError("OLR received an old or repeated optimizer iteration")
        epoch = (step - 1) // len(self.trainer.train_dataloader)
        groups = defaultdict(list)
        for i, uid in enumerate(data.non_tensor_batch["uid"]):
            groups[uid].append(i)
        observations = []
        diagnostic_inputs = []
        extracted_count = 0
        previous = deepcopy(self.refiner.state_dict())
        # Repeated responses can share dictionaries; only change private copies.
        old_extras = data.non_tensor_batch["extra_info"].copy()
        old_targets = data.non_tensor_batch["reward_model"].copy()
        try:
            for indices in groups.values():
                first = data[indices[0]]
                extra = first.non_tensor_batch["extra_info"]
                task = first.non_tensor_batch["data_source"]
                if task.startswith("val/"):
                    raise RuntimeError("Held-out data entered the training refiner")
                verifier = verifier_classes[task]()
                keys = []
                raw_answers = []
                for i in indices:
                    item = data[i]
                    prompt_length = item.batch["prompts"].shape[-1]
                    length = int(item.batch["attention_mask"][prompt_length:].sum())
                    text = self.reward.tokenizer.decode(item.batch["responses"][:length], skip_special_tokens=True)
                    extracted = extract_answer(text)
                    raw_answers.append(extracted)
                    try:
                        with verifier_deadline():
                            key = answer_key(extracted, verifier)
                    except (Exception, TaskDeadlineExceeded):
                        key = None
                    keys.append(key)
                    extracted_count += key is not None
                decision = self.refiner.observe(extra["index"], keys, epoch)
                observations.append(decision)
                diagnostic_inputs.append({
                    "prompt_id": int(extra["index"]), "task": task,
                    "original_noisy": extra["target_mode"] == "fixed_wrong_answer",
                    "original_assigned_answer_key": extra.get("assigned_answer_key"),
                    "original_game_data_str": extra.get("original_game_data_str") or extra["game_data_str"],
                    "raw_answers": raw_answers, "answer_keys": keys, **decision})
                if decision["selected"]:
                    for i in indices:
                        current = deepcopy(data.non_tensor_batch["extra_info"][i])
                        current.update(target_mode="fixed_wrong_answer", assigned_answer_key=decision["answer_key"])
                        data.non_tensor_batch["extra_info"][i] = current
                        target = deepcopy(data.non_tensor_batch["reward_model"][i])
                        target["ground_truth"] = decision["answer_key"]
                        data.non_tensor_batch["reward_model"][i] = target
            result = self.reward(data, return_dict=True)
            for record, indices in zip(diagnostic_inputs, groups.values()):
                record["training_rewards"] = [float(result["reward_tensor"][i].sum()) for i in indices]
            def verify_original(task, game, answer):
                try:
                    with verifier_deadline():
                        return verifier_classes[task]().verify(Data.from_json_str(game), answer)
                except TaskDeadlineExceeded as error:
                    raise TimeoutError("Native diagnostic verifier deadline") from error
            diagnostic_records, diagnostic_metrics = diagnose(diagnostic_inputs, verify_original)
            prompts = {str(row["prompt_id"]): {key: row[key] for key in (
                "task", "original_noisy", "original_assigned_answer_key", "original_game_data_str")}
                for row in diagnostic_inputs}
            for prompt_id, details in prompts.items():
                if prompt_id in self.correctness["prompts"] and self.correctness["prompts"][prompt_id] != details:
                    raise ValueError("Original diagnostic labels changed during training")
            if self.correctness["through_step"] != step - 1:
                raise RuntimeError("Diagnostic history and optimizer position disagree")
            batch = {"training_update": step, "policy_update": step - 1, "epoch": epoch,
                     "diagnostic_implementation_sha256": self.diagnostic_sha256,
                     "allocation_id": os.environ.get("SYNLOGIC_ALLOCATION_ID"),
                     "attempt": os.environ.get("SYNLOGIC_ATTEMPT"),
                     "records": diagnostic_records, "metrics": diagnostic_metrics}
            directory = Path(self.trainer.config.trainer.default_local_dir).parent / "early_correctness"
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / f"step_{step}.json.tmp"
            temporary.write_text(json.dumps({**batch, "prompts": prompts}) + "\n")
            temporary.replace(directory / f"step_{step}.json")
            self.correctness["batches"].append(batch)
            self.correctness["prompts"].update(prompts)
            self.correctness["through_step"] = step
        except BaseException:
            self.refiner.load_state_dict(previous)
            raise
        finally:
            data.non_tensor_batch["extra_info"] = old_extras
            data.non_tensor_batch["reward_model"] = old_targets
        self.last_step = step
        count = len(observations)
        self.metrics = {
            "olr/selected_fraction": sum(o["selected"] for o in observations) / count,
            "olr/mean_slope": sum(o["slope"] for o in observations) / count,
            "olr/majority_rate": sum(o["majority_rate"] for o in observations) / count,
            "olr/mean_visits": sum(o["visits"] for o in observations) / count,
            "olr/unique_prompts_seen": len(self.refiner.history), "olr/epoch": epoch,
            "olr/extractable_response_fraction": extracted_count / len(data),
            "olr/train_reward_mean": float(result["reward_tensor"].sum(-1).mean()),
            "olr/informative_group_fraction": sum(
                len(set(float(result["reward_tensor"][i].sum()) for i in indices)) > 1
                for indices in groups.values()) / count,
        }
        self.cached_batch, self.cached_reward = data, result
        self.metrics.update(diagnostic_metrics)
        return result if return_dict else result["reward_tensor"]


class SynLogicOLRTrainer(RayDAPOTrainer):
    def _create_dataloader(self, *args, **kwargs):
        super()._create_dataloader(*args, **kwargs)
        from torch.utils.data import RandomSampler
        from .sampling import ResumableDataLoader, ResumableRandomSampler

        original = self.train_dataloader
        if not isinstance(original.sampler, RandomSampler):
            raise ValueError("SynLogic checkpoint recovery requires random sampling")
        sampler = ResumableRandomSampler(original.dataset,
            replacement=original.sampler.replacement,
            num_samples=original.sampler.num_samples,
            generator=original.sampler.generator)
        self.train_dataloader = ResumableDataLoader(
            dataset=original.dataset, batch_size=original.batch_size,
            num_workers=original.num_workers, drop_last=original.drop_last,
            collate_fn=original.collate_fn, sampler=sampler,
            legacy_sampler_epoch=self.config.data.get("legacy_sampler_epoch"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.reward_fn, SynLogicRewardManager):
            raise TypeError("The SynLogic refiner requires the native puzzle reward manager")
        self.reward_fn = RefinementReward(self, self.reward_fn)
        self.val_reward_fn = ValidationRecorder(self.val_reward_fn)

    def _save_checkpoint(self):
        folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"step": self.global_steps, "last_reward_step": self.reward_fn.last_step,
                   "protocol_id": self.config.provenance.protocol_id,
                   "prompt_template_sha256": self.config.provenance.prompt_template_sha256,
                   "refiner": self.reward_fn.refiner.state_dict(),
                   "early_correctness": self.reward_fn.correctness}
        temporary = folder / "olr_state.json.tmp"
        temporary.write_text(json.dumps(payload) + "\n")
        temporary.replace(folder / "olr_state.json")
        if self.global_steps in {5 * len(self.train_dataloader), int(self.config.trainer.total_training_steps)}:
            (folder / "comparison.json").write_text(json.dumps({"step": self.global_steps,
                                                               "block_updates": int(self.config.trainer.total_training_steps)}) + "\n")
        # The native complete marker and recovery pointer are published only
        # after the refinement state and every model/optimizer shard exist.
        super()._save_checkpoint()

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        if self.global_steps:
            folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
            state = json.loads((folder / "olr_state.json").read_text())
            if (state.get("protocol_id") != self.config.provenance.protocol_id
                    or state.get("prompt_template_sha256") != self.config.provenance.prompt_template_sha256):
                raise ValueError("Checkpoint prompt protocol differs from the corrected comparison")
            if state["step"] != self.global_steps or state["last_reward_step"] != self.global_steps:
                raise ValueError("Model and OLR histories belong to different updates")
            self.reward_fn.refiner.load_state_dict(state["refiner"])
            self.reward_fn.last_step = state["last_reward_step"]
            diagnostic = state.get("early_correctness", {"version": 1,
                "missing_through_step": self.global_steps, "through_step": self.global_steps,
                "prompts": {}, "batches": []})
            if diagnostic["version"] != 1 or diagnostic["through_step"] != self.global_steps:
                raise ValueError("Incompatible true-correctness diagnostic history")
            if [b["training_update"] for b in diagnostic["batches"]] != list(
                    range(diagnostic["missing_through_step"] + 1, self.global_steps + 1)):
                raise ValueError("Diagnostic coverage has unrecorded internal updates")
            self.reward_fn.correctness = diagnostic
        return result

    def _validate(self):
        self.val_reward_fn.records.clear()
        metrics = super()._validate()
        values = [v for k, v in metrics.items() if k.startswith("val-core/val/")
                  and k.endswith("/reward/mean@8")]
        if len(values) != 27:
            raise ValueError(f"Expected all 27 held-out puzzle tasks, received {len(values)}")
        metrics["val-core/synlogic/reward/mean@8"] = sum(values) / len(values)
        overlap = set(self.config.evaluation.overlap_indices)
        if overlap:
            raise ValueError("Training must exclude all held-out evaluation examples")
        scores = [score for index, score in self.val_reward_fn.records if index not in overlap]
        if len(scores) != (270 - len(overlap)) * 8:
            raise ValueError("Disjoint validation rows or response counts are incomplete")
        metrics["val-core/synlogic-disjoint/reward/mean@8"] = sum(scores) / len(scores)
        invalid = set(self.config.evaluation.invalid_indices)
        audited = [score for index, score in self.val_reward_fn.records if index not in invalid]
        if len(invalid) != 16 or len(audited) != 254 * 8:
            raise ValueError("Audited validation rows or response counts are incomplete")
        metrics["val-core/synlogic-audited/reward/mean@8"] = sum(audited) / len(audited)
        return metrics

    def fit(self):
        import verl.utils.tracking as tracking
        original = tracking.Tracking
        trainer = self

        class RefinementTracking(original):
            def log(self, data, step):
                if trainer.reward_fn.last_step == step:
                    data.update(trainer.reward_fn.metrics)
                return super().log(data, step)

        tracking.Tracking = RefinementTracking
        try:
            return super().fit()
        finally:
            tracking.Tracking = original
