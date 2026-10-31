"""Puzzle-domain OLR integration with complete refinement-state recovery."""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

from src.dapo_ray_trainer import RayDAPOTrainer
from src.fixed_labels import answer_key, extract_answer
from src.syn_logic_reward import SynLogicRewardManager, verifier_classes, verifier_deadline

from .refinement import OnlineLabelRefiner


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
                for i in indices:
                    item = data[i]
                    prompt_length = item.batch["prompts"].shape[-1]
                    length = int(item.batch["attention_mask"][prompt_length:].sum())
                    text = self.reward.tokenizer.decode(item.batch["responses"][:length], skip_special_tokens=True)
                    extracted = extract_answer(text)
                    try:
                        with verifier_deadline():
                            key = answer_key(extracted, verifier)
                    except Exception:
                        key = None
                    keys.append(key)
                    extracted_count += key is not None
                decision = self.refiner.observe(extra["index"], keys, epoch)
                observations.append(decision)
                if decision["selected"]:
                    for i in indices:
                        current = deepcopy(data.non_tensor_batch["extra_info"][i])
                        current.update(target_mode="fixed_wrong_answer", assigned_answer_key=decision["answer_key"])
                        data.non_tensor_batch["extra_info"][i] = current
                        target = deepcopy(data.non_tensor_batch["reward_model"][i])
                        target["ground_truth"] = decision["answer_key"]
                        data.non_tensor_batch["reward_model"][i] = target
            result = self.reward(data, return_dict=True)
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
        return result if return_dict else result["reward_tensor"]


class SynLogicOLRTrainer(RayDAPOTrainer):
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
                   "refiner": self.reward_fn.refiner.state_dict()}
        temporary = folder / "olr_state.json.tmp"
        temporary.write_text(json.dumps(payload) + "\n")
        temporary.replace(folder / "olr_state.json")
        if self.global_steps % 970 == 0:
            (folder / "comparison.json").write_text(json.dumps({"step": self.global_steps,
                                                               "block_updates": 970}) + "\n")
        # The native complete marker and recovery pointer are published only
        # after the refinement state and every model/optimizer shard exist.
        super()._save_checkpoint()

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        if self.global_steps:
            folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
            state = json.loads((folder / "olr_state.json").read_text())
            if state["step"] != self.global_steps or state["last_reward_step"] != self.global_steps:
                raise ValueError("Model and OLR histories belong to different updates")
            self.reward_fn.refiner.load_state_dict(state["refiner"])
            self.reward_fn.last_step = state["last_reward_step"]
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
        if len(overlap) != 36:
            raise ValueError("The disjoint metric requires the original 36-question overlap audit")
        scores = [score for index, score in self.val_reward_fn.records if index not in overlap]
        if len(scores) != (270 - len(overlap)) * 8:
            raise ValueError("Disjoint validation rows or response counts are incomplete")
        metrics["val-core/synlogic-disjoint/reward/mean@8"] = sum(scores) / len(scores)
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
