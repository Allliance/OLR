"""Observe native true-label correctness without supplying it to RL or OLR."""

from contextlib import redirect_stderr, redirect_stdout
import io
import random
import time

import numpy as np
import torch


def diagnose(records, verify, budget_seconds=30):
    """Score original extracted answers; retain unknowns and bound CPU overhead.

    ``verify(task, original_game_json, raw_answer)`` must impose its own per-call
    deadline. Canonical keys define the refiner's majority, not verifier inputs.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    started = time.monotonic()
    output = []
    try:
        for original in records:
            record = dict(original)
            game = record.pop("original_game_data_str")
            cache, errors = {}, {}
            for answer in record["raw_answers"]:
                if answer in cache:
                    continue
                if answer is None:
                    cache[answer] = False
                elif time.monotonic() - started >= budget_seconds:
                    cache[answer], errors[answer] = None, "batch_verification_budget"
                else:
                    try:
                        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            verdict = verify(record["task"], game, answer)
                            if verdict not in (True, False, 0, 1):
                                raise ValueError("Native verifier returned a nonbinary verdict")
                            cache[answer] = bool(verdict)
                    except Exception as error:
                        cache[answer], errors[answer] = None, type(error).__name__
            scores = [cache[answer] for answer in record["raw_answers"]]
            majority = record["answer_key"]
            members = [score for key, score in zip(record["answer_keys"], scores)
                       if majority is not None and key == majority]
            if majority is None:
                correct = False
            elif members and all(score is True for score in members):
                correct = True
            elif members and all(score is False for score in members):
                correct = False
            else:
                # A lossy canonical class or timeout cannot be called correct.
                correct = None
            record.update(native_rollout_correct=scores, native_majority_correct=correct,
                          verification_errors=[errors.get(answer) for answer in record["raw_answers"]])
            output.append(record)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)

    metrics = {"early_correctness/verification_seconds": time.monotonic() - started}
    for name, noisy in (("clean", False), ("noisy", True)):
        group = [row for row in output if row["original_noisy"] is noisy]
        if not group:
            continue
        prefix = f"early_correctness/{name}/"
        correct = sum(row["native_majority_correct"] is True for row in group)
        unknown = sum(row["native_majority_correct"] is None for row in group)
        metrics.update({prefix + "prompts": len(group), prefix + "majority_correct": correct,
                        prefix + "majority_unverified": unknown,
                        prefix + "majority_accuracy_lower_bound": correct / len(group),
                        prefix + "majority_accuracy_upper_bound": (correct + unknown) / len(group)})
        if unknown < len(group):
            metrics[prefix + "majority_accuracy_known"] = correct / (len(group) - unknown)
        rollouts = [score for row in group for score in row["native_rollout_correct"]]
        successes, unverified = sum(s is True for s in rollouts), sum(s is None for s in rollouts)
        metrics[prefix + "rollout_accuracy_lower_bound"] = successes / len(rollouts)
        metrics[prefix + "rollout_accuracy_upper_bound"] = (successes + unverified) / len(rollouts)
        selected = [row for row in group if row["selected"]]
        metrics[prefix + "selected_prompts"] = len(selected)
        if selected:
            good = sum(row["native_majority_correct"] is True for row in selected)
            unsure = sum(row["native_majority_correct"] is None for row in selected)
            metrics[prefix + "selected_accuracy_lower_bound"] = good / len(selected)
            metrics[prefix + "selected_accuracy_upper_bound"] = (good + unsure) / len(selected)
        if noisy:
            hits = sum(key == row["original_assigned_answer_key"]
                       for row in group for key in row["answer_keys"])
            metrics[prefix + "original_wrong_target_hit_rate"] = hits / len(rollouts)
    return output, metrics
