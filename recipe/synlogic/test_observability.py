"""Check mask definitions, zero-denominator semantics and complete snapshot replay."""

import copy
import gzip
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .observability import (commit_snapshot, file_digest, noisy_metrics, plain, snapshot_due,
                            snapshot_payload, token_digest, validate_snapshot, write_snapshot)


def fixture(real=False):
    uids = ["clean", "clean", "n1", "n1", "n2", "n2"]
    responses = torch.tensor([[101+i, 201+i, 0] for i in range(6)])
    mask = torch.tensor([[1, 1, 0]] * 6)
    advantages = torch.tensor([[1., 0., 7.], [0., 1., 7.], [0., 0., 7.],
                               [0., 0., 7.], [0., 1., 7.], [0., 0., 7.]])
    records, registry = [], {}
    for j, uid in enumerate(("clean", "n1", "n2")):
        start = 2*j
        records.append({"native_group_uid": uid, "prompt_id": j, "task": "toy",
            "original_noisy": uid != "clean", "original_assigned_answer_key": "wrong",
            "raw_answers": ["first", "second"], "answer_keys": ["wrong", "other" if uid != "n2" else None],
            "canonical_key_errors": [None, "TimeoutError" if uid == "n2" else None],
            "native_rollout_correct": [None if uid == "n2" else True, uid == "n2"],
            "verification_errors": ["TimeoutError" if uid == "n2" else None, None],
            "response_token_sha256": [token_digest(responses[k][mask[k].bool()]) for k in (start, start+1)],
            "selected": False, "answer_key": "wrong", "slope": 0., "majority_rate": .5, "visits": 1})
        registry[str(j)] = {"task": "toy", "original_noisy": uid != "clean",
            "original_assigned_answer_key": "wrong",
            "original_game_data_str": json.dumps({"question": f"question {j}", "answer": "gold"})}
    # Rebalancing may permute both groups and responses after reward calculation.
    permutation = torch.tensor([4, 0, 3, 1, 5, 2])
    tensors = {"prompts": torch.tensor([[0, 10+i//2] for i in range(6)]), "responses": responses,
        "attention_mask": torch.tensor([[0, 1, 1, 1, 0]]*6), "response_mask": mask,
        "advantages": advantages, "token_level_rewards": torch.zeros(6, 3),
        "token_level_scores": torch.zeros(6, 3), "position_ids": torch.arange(5).repeat(6, 1),
        "old_log_probs": torch.full((6, 3), -.5)}
    batch = SimpleNamespace(batch={key: value[permutation].clone() for key, value in tensors.items()},
        non_tensor_batch={"uid": np.array([uids[i] for i in permutation], dtype=object),
                          "reward_model": [{"ground_truth": "frozen"} for _ in permutation]})
    if real:
        from tensordict import TensorDict
        from verl import DataProto
        batch = DataProto(batch=TensorDict(batch.batch, batch_size=[6]),
                          non_tensor_batch={key: np.array(value, dtype=object) for key, value in batch.non_tensor_batch.items()})
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: ",".join(map(str, ids.tolist())))
    metadata = {"protocol_id": "toy", "training_update": 19, "policy_update": 18,
                "allocation_id": "cpu-fixture", "attempt": "unit-test", "configuration": {"lr": 1e-6}}
    return batch, records, registry, tokenizer, metadata


class ObservabilityTests(unittest.TestCase):
    def test_trainer_capture_and_optimizer_commit_are_separate(self):
        from omegaconf import OmegaConf
        from .trainer import SynLogicOLRTrainer
        batch, records, registry, tokenizer, _ = fixture(real=True)
        original = {key: value.clone() for key, value in batch.batch.items()}
        with tempfile.TemporaryDirectory() as directory:
            subject = object.__new__(SynLogicOLRTrainer)
            subject.config = OmegaConf.create({"trainer": {"default_local_dir": directory + "/checkpoints", "test_freq": 20},
                "provenance": {"protocol_id": "cpu-test"}})
            subject.global_steps, subject.total_training_steps = 19, 880
            subject._observation_resume_step = 18
            subject._progress = lambda *_: None
            subject.reward_fn = SimpleNamespace(correctness={"batches": [{"training_update": 19, "records": records}],
                "prompts": registry}, reward=SimpleNamespace(tokenizer=tokenizer))
            with patch.dict("os.environ", {"SYNLOGIC_ATTEMPT": "cpu-test"}):
                metrics = subject._observe_training_batch(batch)
                self.assertEqual(len(metrics), 21)
                path = Path(directory)/"training_rollouts/attempt_cpu-test/snapshots/global_step_19"
                self.assertTrue((path/"complete.json").exists())
                self.assertFalse((path/"committed.json").exists())
                subject._training_update_observed(batch)
                self.assertTrue((path/"committed.json").exists())
                self.assertEqual(validate_snapshot(path)["step"], 19)
            for key, value in original.items():
                self.assertTrue(torch.equal(batch.batch[key], value))

    def test_native_reward_status_is_per_rollout_without_changing_scores(self):
        from src.syn_logic_reward import SynLogicRewardManager
        batch, *_ = fixture(real=True)
        batch.non_tensor_batch["data_source"] = np.array(["toy"]*6, dtype=object)
        batch.non_tensor_batch["extra_info"] = np.array([{}]*6, dtype=object)
        tokenizer = SimpleNamespace(eos_token="<eos>", decode=lambda ids, **_: str(ids.tolist()))
        subject = SynLogicRewardManager(tokenizer, 0, max_resp_len=3,
            overlong_buffer_cfg=SimpleNamespace(enable=False))
        count = []
        def score(**kwargs):
            i = len(count)
            kwargs["extra_info"]["verifier_timed_out"] = i == 1
            kwargs["extra_info"]["verifier_unverified"] = i == 4
            count.append(i)
            return float(i % 2)
        subject.compute_score = score
        result = subject(batch, return_dict=True)
        self.assertEqual(result["reward_tensor"].sum(-1).tolist(), [0., 1., 0., 1., 0., 1.])
        self.assertEqual(result["reward_extra_info"]["training_verifier_timed_out"], [False, True, False, False, False, False])
        self.assertEqual(result["reward_extra_info"]["training_verifier_unverified"], [False, False, False, False, True, False])
        self.assertTrue(all(not info for info in batch.non_tensor_batch["extra_info"]))

    def test_actual_masked_advantages_and_coverage(self):
        batch, records, *_ = fixture()
        metrics = noisy_metrics(batch.batch["advantages"], batch.batch["response_mask"],
                                batch.non_tensor_batch["uid"], records)
        self.assertEqual(metrics["noise/noisy_group_active/numerator"], 1)
        self.assertEqual(metrics["noise/noisy_group_active/denominator"], 2)
        self.assertEqual(metrics["noise/noisy_group_active/rate"], .5)
        for name in ("assigned_label_match", "original_verifier_correctness"):
            prefix = "noise/noisy_rollout_" + name
            self.assertEqual(metrics[prefix+"/numerator"], 2)
            self.assertEqual(metrics[prefix+"/denominator"], 3)
            self.assertEqual(metrics[prefix+"/eligible"], 4)
            self.assertEqual(metrics[prefix+"/failures"], 1)
            self.assertEqual(metrics[prefix+"/coverage"], .75)

    def test_zero_denominators_are_undefined(self):
        batch, records, *_ = fixture()
        for row in records:
            row["original_noisy"] = False
        metrics = noisy_metrics(batch.batch["advantages"], batch.batch["response_mask"],
                                batch.non_tensor_batch["uid"], records)
        for name in ("noisy_group_active", "noisy_rollout_assigned_label_match", "noisy_rollout_original_verifier_correctness"):
            prefix = "noise/" + name
            self.assertIsNone(metrics[prefix+"/rate"])
            self.assertIsNone(metrics[prefix+"/coverage"])
            self.assertEqual(metrics[prefix+"/rate_defined"], 0)
        json.dumps(metrics, allow_nan=False)

    def test_nonfinite_valid_advantage_is_not_counted_active(self):
        batch, records, *_ = fixture()
        batch.batch["advantages"][0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            noisy_metrics(batch.batch["advantages"], batch.batch["response_mask"], batch.non_tensor_batch["uid"], records)

    def test_snapshot_precedes_evaluation(self):
        self.assertTrue(snapshot_due(19, 20, 880))
        self.assertTrue(snapshot_due(879, 20, 880))
        self.assertFalse(snapshot_due(20, 20, 880))
        self.assertFalse(snapshot_due(0, 20, 880))
        self.assertFalse(snapshot_due(880, 20, 880))

    def test_full_snapshot_is_lossless_atomic_committed_and_replay_safe(self):
        batch, records, registry, tokenizer, metadata = fixture()
        original = {key: value.clone() for key, value in batch.batch.items()}
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
        with tempfile.TemporaryDirectory() as directory:
            payload = snapshot_payload(batch, records, registry, tokenizer, metadata)
            path = write_snapshot(directory, 19, payload)
            self.assertFalse((Path(directory)/"latest_committed_snapshot.txt").exists())
            commit_snapshot(path, 19)
            validated = validate_snapshot(path)
            self.assertEqual(validated["rollouts"], 6)
            self.assertEqual(validated["groups"], 3)
            with gzip.open(path/"rollouts.pt.gz", "rb") as stream:
                restored = torch.load(stream, weights_only=True)
            for key, value in original.items():
                self.assertTrue(torch.equal(restored["tensors"][key], value))
                self.assertTrue(torch.equal(batch.batch[key], value))
            self.assertEqual({row["generation_index"] for row in restored["rollouts"]}, {0, 1})
            before = file_digest(path/"rollouts.pt.gz")
            self.assertEqual(write_snapshot(directory, 19, payload), path)
            self.assertEqual(file_digest(path/"rollouts.pt.gz"), before)
            different = copy.deepcopy(payload)
            different["tensors"]["responses"][0, 0] += 1
            with self.assertRaises(FileExistsError):
                write_snapshot(directory, 19, different)
            self.assertEqual(file_digest(path/"rollouts.pt.gz"), before)
        self.assertEqual(random.getstate(), python_state)
        after = np.random.get_state()
        np.testing.assert_array_equal(after[1], numpy_state[1])
        self.assertEqual(after[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_provenance_redacts_credentials_preserving_token_configuration(self):
        result = plain({"api_key": "secret-value", "token_file": "/private/path", "max_prompt_tokens": 24576})
        self.assertEqual(result["api_key"], "[REDACTED]")
        self.assertEqual(result["token_file"], "[REDACTED]")
        self.assertEqual(result["max_prompt_tokens"], 24576)


if __name__ == "__main__":
    unittest.main()
