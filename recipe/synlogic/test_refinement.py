"""Regression checks for the scientific gates and resumed history."""

import copy
import importlib.util
import json
from pathlib import Path
import random
import unittest

spec = importlib.util.spec_from_file_location("refinement", Path(__file__).with_name("refinement.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
OnlineLabelRefiner = module.OnlineLabelRefiner


class RefinementTests(unittest.TestCase):
    def test_warmup_then_positive_consistent_trajectory(self):
        refiner = OnlineLabelRefiner(True)
        for epoch, count in enumerate((2, 3, 4, 5, 6, 7)):
            decision = refiner.observe(1, ["a"] * count + [f"other-{i}" for i in range(8-count)], epoch)
            self.assertEqual(decision["selected"], epoch > 4)
        self.assertAlmostEqual(decision["slope"], 0.125)

    def test_current_and_historical_ties_rejected(self):
        refiner = OnlineLabelRefiner(True, start_select_epoch=-1, slope_threshold=-1)
        self.assertFalse(refiner.observe(1, ["a", "b"], 9)["selected"])
        refiner.observe(2, ["a", "a", "b"], 9)
        self.assertFalse(refiner.observe(2, ["b", "b", "a"], 9)["selected"])

    def test_nonincreasing_trajectories_rejected(self):
        for counts in ((6, 5, 4, 3, 2, 2), (4, 4, 4, 4, 4, 4)):
            refiner = OnlineLabelRefiner(True)
            for epoch, count in enumerate(counts):
                decision = refiner.observe(1, ["a"] * count + [f"x{i}" for i in range(8-count)], epoch)
            self.assertFalse(decision["selected"])

    def test_missing_answers_never_become_labels(self):
        refiner = OnlineLabelRefiner(True, start_select_epoch=-1, slope_threshold=-1)
        decision = refiner.observe(1, [None] * 8, 10)
        self.assertFalse(decision["selected"])
        self.assertEqual(decision["majority_rate"], 0)
        self.assertEqual(refiner.observe(2, [None]*6 + ["a", "a"], 10)["majority_rate"], 0.25)
        self.assertFalse(refiner.observe(3, [None]*7 + ["a"], 10)["selected"])

    def test_baseline_never_changes_labels(self):
        refiner = OnlineLabelRefiner(False, start_select_epoch=-1, slope_threshold=-1)
        self.assertFalse(refiner.observe(1, ["a"] * 8, 100)["selected"])

    def test_recovery_preserves_future_decisions(self):
        first = OnlineLabelRefiner(True)
        for epoch, count in enumerate((2, 3, 4, 5, 6)):
            first.observe(4, ["a"] * count + [None] * (8-count), epoch)
        restored = OnlineLabelRefiner(True)
        restored.load_state_dict(json.loads(json.dumps(first.state_dict())))
        self.assertEqual(first.observe(4, ["a"]*7+[None], 5), restored.observe(4, ["a"]*7+[None], 5))

    def test_recovery_rejects_changed_protocol_or_broken_history(self):
        refiner = OnlineLabelRefiner(True)
        refiner.observe(1, ["a"], 0)
        for corrupt in ("config", "history"):
            state = copy.deepcopy(refiner.state_dict())
            if corrupt == "config": state["config"]["slope_threshold"] = 0.1
            else: state["history"]["1"]["rates"] = [float("nan")]
            with self.assertRaises(ValueError): OnlineLabelRefiner(True).load_state_dict(state)

    def test_slope_matches_independent_pairwise_ols(self):
        rng = random.Random(42)
        for length in range(2, 50):
            y = [rng.random() for _ in range(length)]
            numerator = sum((j-i)*(y[j]-y[i]) for i in range(length) for j in range(i+1,length))
            denominator = sum((j-i)**2 for i in range(length) for j in range(i+1,length))
            self.assertAlmostEqual(module.trajectory_slope(y), numerator / denominator, places=14)


if __name__ == "__main__":
    unittest.main()
