"""The upstream OLR slope and historical-majority gates, without an oracle.

Adapted from RayPPOTrainer._generate_pseudo_labels_and_update_passrate_weak_noise_label
and _get_slope in this repository. Puzzle answers are canonical keys rather than
math-verify expressions. Fixed assigned labels are never regenerated online.
"""

from collections import Counter
import math


def unique_majority(values):
    counts = Counter(values)
    if not counts:
        return None
    highest = max(counts.values())
    winners = [key for key, count in counts.items() if count == highest]
    return winners[0] if len(winners) == 1 else None


def trajectory_slope(values):
    if len(values) < 2:
        return 0.0
    center = (len(values) - 1) / 2
    mean = math.fsum(values) / len(values)
    return math.fsum((i - center) * (v - mean) for i, v in enumerate(values)) / math.fsum(
        (i - center) ** 2 for i in range(len(values))
    )


class OnlineLabelRefiner:
    def __init__(self, enabled, start_select_epoch=4, slope_threshold=0.05):
        self.config = {"enabled": bool(enabled), "start_select_epoch": start_select_epoch,
                       "slope_threshold": slope_threshold}
        self.history = {}

    def observe(self, prompt_id, answer_keys, epoch):
        if not answer_keys:
            raise ValueError("An OLR observation must include all generated responses")
        # An unextractable response cannot become an answer target. It still
        # occupies one rollout in the pass-rate denominator.
        majority = unique_majority([key for key in answer_keys if key is not None])
        # Upstream's unparseable responses remain distinct singleton groups.
        # A lone extractable answer therefore cannot win a frequency-one tie.
        if majority is not None and answer_keys.count(majority) == 1 and None in answer_keys:
            majority = None
        rate = answer_keys.count(majority) / len(answer_keys) if majority is not None else 0.0
        history = self.history.setdefault(str(prompt_id), {"answers": [], "rates": []})
        history["answers"].append(majority)
        history["rates"].append(rate)
        slope = trajectory_slope(history["rates"])
        historical = unique_majority(history["answers"])
        selected = (self.config["enabled"] and epoch > self.config["start_select_epoch"]
                    and majority is not None and historical == majority
                    and slope >= self.config["slope_threshold"])
        return {"selected": selected, "answer_key": majority, "majority_rate": rate,
                "slope": slope, "visits": len(history["rates"])}

    def state_dict(self):
        return {"version": 1, "config": self.config, "history": self.history}

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("config") != self.config:
            raise ValueError("Incompatible OLR recovery configuration")
        for key, history in state["history"].items():
            if not isinstance(key, str) or len(history["answers"]) != len(history["rates"]):
                raise ValueError("Incomplete OLR history")
            if not all(answer is None or isinstance(answer, str) for answer in history["answers"]):
                raise ValueError("Invalid OLR answer history")
            if not all(isinstance(rate, (int, float)) and math.isfinite(rate) and 0 <= rate <= 1
                       for rate in history["rates"]):
                raise ValueError("Invalid OLR pass-rate history")
        self.history = state["history"]
