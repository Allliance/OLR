"""Guard diagnostic correctness, uncertainty and isolation from training."""

import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .diagnostics import diagnose


def record():
    return {"prompt_id": 1, "task": "test_task", "original_noisy": True,
            "original_assigned_answer_key": "wrong", "original_game_data_str": "original",
            "raw_answers": ["raw-valid", "raw-invalid", None],
            "answer_keys": ["same-key", "same-key", None],
            "answer_key": "same-key", "selected": True, "slope": 0.1, "visits": 6}


class DiagnosticsTests(unittest.TestCase):
    def test_checkpoint_coverage_migrates_and_rejects_a_gap(self):
        from . import trainer as module
        from .refinement import OnlineLabelRefiner
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "global_step_5"
            folder.mkdir()
            subject = object.__new__(module.SynLogicOLRTrainer)
            subject.global_steps = 5
            subject.config = SimpleNamespace(trainer=SimpleNamespace(default_local_dir=directory),
                provenance=SimpleNamespace(protocol_id="protocol", prompt_template_sha256="template"))
            subject.reward_fn = SimpleNamespace(refiner=OnlineLabelRefiner(True), last_step=0)
            state = {"step": 5, "last_reward_step": 5, "protocol_id": "protocol",
                     "prompt_template_sha256": "template", "refiner": subject.reward_fn.refiner.state_dict()}
            path = folder / "olr_state.json"
            with patch.object(module.RayDAPOTrainer, "_load_checkpoint", return_value=None):
                path.write_text(json.dumps(state))
                subject._load_checkpoint()
                self.assertEqual(subject.reward_fn.correctness,
                    {"version": 1, "missing_through_step": 5, "through_step": 5, "prompts": {}, "batches": []})
                state["early_correctness"] = {"version": 1, "missing_through_step": 3, "through_step": 5,
                    "prompts": {}, "batches": [{"training_update": 4}, {"training_update": 5}]}
                path.write_text(json.dumps(state))
                subject._load_checkpoint()
                self.assertEqual(subject.reward_fn.correctness, state["early_correctness"])
                state["early_correctness"]["batches"].pop(0)
                path.write_text(json.dumps(state))
                with self.assertRaises(ValueError):
                    subject._load_checkpoint()

    def test_raw_answers_and_uncertain_canonical_class(self):
        source = record()
        original = copy.deepcopy(source)
        def verify(task, game, answer):
            self.assertEqual(game, "original")
            self.assertNotEqual(answer, "same-key")
            return answer == "raw-valid"
        rows, metrics = diagnose([source], verify)
        self.assertEqual(source, original)
        self.assertEqual(rows[0]["native_rollout_correct"], [True, False, False])
        self.assertIsNone(rows[0]["native_majority_correct"])
        self.assertEqual(metrics["early_correctness/noisy/majority_accuracy_lower_bound"], 0)
        self.assertEqual(metrics["early_correctness/noisy/majority_accuracy_upper_bound"], 1)

    def test_verifier_failure_and_budget_remain_unknown(self):
        def fail(*args):
            raise TimeoutError()
        for budget in (0, 30):
            rows, metrics = diagnose([record()], fail, budget_seconds=budget)
            self.assertEqual(rows[0]["native_rollout_correct"], [None, None, False])
            self.assertIsNone(rows[0]["native_majority_correct"])
            self.assertEqual(metrics["early_correctness/noisy/majority_unverified"], 1)

    def test_training_reward_and_refinement_ignore_diagnostic_oracle(self):
        import torch
        from . import trainer as module

        class Oracle:
            truth = False
            def extract_answer(self, answer):
                return answer
            def verify(self, game, answer):
                assert game.answer == "gold", "Corrupted labels leaked into diagnostics"
                random.random()
                np.random.random()
                torch.rand(1)
                return self.truth

        class Batch:
            def __init__(self):
                self.non_tensor_batch = {
                    "uid": np.array(["uid"] * 8, dtype=object),
                    "extra_info": np.array([{"index": 1, "target_mode": "fixed_wrong_answer",
                        "assigned_answer_key": "wrong", "game_data_str": json.dumps({"question": "q", "answer": "wrong"}),
                        "original_game_data_str": json.dumps({"question": "q", "answer": "gold"})}
                        for _ in range(8)], dtype=object),
                    "reward_model": np.array([{"ground_truth": "wrong"} for _ in range(8)], dtype=object)}
            def __len__(self):
                return 8
            def __getitem__(self, index):
                return SimpleNamespace(batch={"prompts": torch.tensor([1]), "attention_mask": torch.tensor([1, 1]),
                    "responses": torch.tensor([index])}, non_tensor_batch={
                    **{k: v[index] for k, v in self.non_tensor_batch.items()}, "data_source": "test_task"})

        class Reward:
            tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: "<answer>" +
                ("a" if int(tokens[0]) < 7 else "b") + "</answer>")
            def __call__(self, data, return_dict=False):
                values = [float(("a" if i < 7 else "b") == extra["assigned_answer_key"])
                          for i, extra in enumerate(data.non_tensor_batch["extra_info"])]
                return {"reward_tensor": torch.tensor(values).reshape(8, 1), "reward_extra_info": {}}

        results = []
        with tempfile.TemporaryDirectory() as directory, patch.dict(module.verifier_classes, {"test_task": Oracle}):
            for truth in (False, True):
                Oracle.truth = truth
                trainer = SimpleNamespace(global_steps=6, train_dataloader=[0], config=SimpleNamespace(
                    olr={"enabled": True, "start_select_epoch": 4, "slope_threshold": .05},
                    trainer=SimpleNamespace(default_local_dir=str(Path(directory) / "checkpoints"))))
                reward = module.RefinementReward(trainer, Reward())
                for epoch, count in enumerate((2, 3, 4, 5, 6)):
                    reward.refiner.observe(1, ["a"] * count + [f"other-{i}" for i in range(8-count)], epoch)
                reward.last_step = 5
                reward.correctness.update(missing_through_step=5, through_step=5)
                data = Batch()
                python_before, numpy_before = random.getstate(), np.random.get_state()
                torch_before = torch.random.get_rng_state()
                result = reward(data, return_dict=True)
                self.assertEqual(random.getstate(), python_before)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))
                after = np.random.get_state()
                self.assertEqual(after[0], numpy_before[0])
                np.testing.assert_array_equal(after[1], numpy_before[1])
                self.assertEqual(after[2:], numpy_before[2:])
                self.assertTrue(all(x["assigned_answer_key"] == "wrong" for x in data.non_tensor_batch["extra_info"]))
                self.assertEqual(reward.correctness["batches"][0]["records"][0]["native_majority_correct"], truth)
                self.assertEqual(reward.correctness["batches"][0]["records"][0]["training_rewards"], [1.0]*7+[0.0])
                self.assertEqual(json.loads(reward.correctness["prompts"]["1"]["original_game_data_str"])["answer"], "gold")
                results.append((result["reward_tensor"], copy.deepcopy(reward.refiner.state_dict())))
            self.assertTrue(torch.equal(results[0][0], results[1][0]))
            self.assertEqual(results[0][1], results[1][1])
            self.assertEqual(results[0][0].sum().item(), 7)


if __name__ == "__main__":
    unittest.main()
