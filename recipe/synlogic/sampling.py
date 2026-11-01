"""Preserve the native RandomSampler stream across later-pass restores."""
from copy import deepcopy

import torch
from torch.utils.data import RandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader


class _Iterator:
    def __init__(self, sampler):
        self.sampler = sampler
        self.start = sampler.generator.get_state().clone()
        self.iterator = RandomSampler.__iter__(sampler)
        self.yielded = 0
        self.exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = next(self.iterator)
        except StopIteration:
            self.exhausted = True
            raise
        self.yielded += 1
        return item

    def state_dict(self):
        return {"version": 1, "shape": self.sampler.shape(),
                "start": self.start.clone(), "yielded": self.yielded,
                "exhausted": self.exhausted,
                "current": self.sampler.generator.get_state().clone()}

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("shape") != self.sampler.shape():
            raise ValueError("Incompatible random-sampler iterator state")
        count = state["yielded"]
        if type(count) is not int or not 0 <= count <= len(self.sampler):
            raise ValueError("Invalid random-sampler cursor")
        self.start = state["start"].clone()
        self.sampler.generator.set_state(self.start)
        self.iterator = RandomSampler.__iter__(self.sampler)
        self.yielded, self.exhausted = 0, False
        for _ in range(count):
            next(self)
        if state["exhausted"]:
            try:
                next(self)
            except StopIteration:
                pass
            else:
                raise ValueError("Sampler marked exhausted before its end")
        if not torch.equal(self.sampler.generator.get_state(), state["current"]):
            raise ValueError("Reconstructed sampler generator differs from checkpoint")


class ResumableRandomSampler(RandomSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.generator is None:
            raise ValueError("Sampler recovery requires an explicit seeded generator")

    def shape(self):
        return {"length": len(self.data_source), "replacement": self.replacement,
                "num_samples": self.num_samples}

    def __iter__(self):
        return _Iterator(self)


def _main(state):
    return state["_snapshot"]["_main_snapshot"] if "_snapshot" in state else state


def _upgrade_legacy(state, sampler, epoch):
    if type(epoch) is not int or epoch < 0:
        raise ValueError("Legacy sampler lacks RNG state; a validated local sampler epoch is required")
    state = deepcopy(state)
    main = _main(state)
    old = main["_sampler_iter_state"]
    if set(old) != {"samples_yielded"} or main.get("_index_sampler_state") is not None:
        raise ValueError("Unsupported legacy sampler state")
    count = old["samples_yielded"]
    if type(count) is not int or not 0 <= count <= sampler.num_samples:
        raise ValueError("Invalid legacy sampler cursor")
    generator = torch.Generator().manual_seed(sampler.generator.initial_seed())
    replay = ResumableRandomSampler(sampler.data_source, replacement=sampler.replacement,
                                   num_samples=sampler.num_samples, generator=generator)
    for _ in range(epoch):
        for _ in RandomSampler.__iter__(replay):
            pass
    iterator = iter(replay)
    for _ in range(count):
        next(iterator)
    main["_sampler_iter_state"] = {"samples_yielded": count,
                                   "sampler_iter_state": iterator.state_dict()}
    if state.get("_iterator_finished"):
        for _ in iterator:
            pass
    state["_synlogic_sampler"] = {"version": 1, "shape": replay.shape(),
                                  "generator": generator.get_state().clone()}
    return state


class ResumableDataLoader(StatefulDataLoader):
    def __init__(self, *args, legacy_sampler_epoch=None, **kwargs):
        self.legacy_sampler_epoch = legacy_sampler_epoch
        super().__init__(*args, **kwargs)
        if not isinstance(self.sampler, ResumableRandomSampler):
            raise ValueError("ResumableDataLoader requires ResumableRandomSampler")

    def state_dict(self):
        state = super().state_dict()
        state["_synlogic_sampler"] = {
            "version": 1, "shape": self.sampler.shape(),
            "generator": self.sampler.generator.get_state().clone()}
        return state

    def load_state_dict(self, state):
        if "_synlogic_sampler" not in state:
            state = _upgrade_legacy(state, self.sampler, self.legacy_sampler_epoch)
        meta = state["_synlogic_sampler"]
        if meta.get("version") != 1 or meta.get("shape") != self.sampler.shape():
            raise ValueError("Incompatible random-sampler checkpoint")
        self.sampler.generator.set_state(meta["generator"])
        state = dict(state)
        del state["_synlogic_sampler"]
        super().load_state_dict(state)
