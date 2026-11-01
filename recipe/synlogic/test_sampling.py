"""Use uninterrupted native sampling as the checkpoint-recovery oracle."""
from copy import deepcopy
import io
import unittest

import torch
from torch.utils.data import Dataset, RandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from .sampling import ResumableDataLoader, ResumableRandomSampler


class Ids(Dataset):
    def __len__(self):
        return 15610

    def __getitem__(self, index):
        return index


def make(fixed, legacy_epoch=None, length=None):
    dataset = Ids() if length is None else list(range(length))
    sampler_class = ResumableRandomSampler if fixed else RandomSampler
    sampler = sampler_class(dataset, generator=torch.Generator().manual_seed(42))
    kwargs = dict(dataset=dataset, batch_size=160, num_workers=4,
                  drop_last=True, sampler=sampler)
    return (ResumableDataLoader(**kwargs, legacy_sampler_epoch=legacy_epoch)
            if fixed else StatefulDataLoader(**kwargs))


def take(loader, count):
    batches = []
    for _ in range(10):
        for batch in loader:
            batches.append(batch.tolist())
            if len(batches) == count:
                return batches
    raise AssertionError('Restored loader did not produce enough batches')


class SamplingRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.points = {20, 97, 150, 194, 198, 231, 312, 399, 445}
        cls.records = {}
        for fixed in [False, True]:
            loader = make(fixed)
            rows, states = [], {}
            for epoch in range(5):
                for batch in loader:
                    rows.append(batch.tolist())
                    if len(rows) in cls.points:
                        states[len(rows)] = deepcopy(loader.state_dict())
                if epoch == 1:
                    exhausted = deepcopy(loader.state_dict())
            cls.records[fixed] = (rows, states, exhausted)

    def test_fresh_order_matches_five_uninterrupted_native_passes(self):
        self.assertEqual(self.records[False][0], self.records[True][0])

    def test_modern_restores_in_later_passes_and_at_boundaries(self):
        rows, states, _ = self.records[True]
        for step, state in states.items():
            with self.subTest(step=step):
                loader = make(True)
                loader.load_state_dict(state)
                self.assertEqual(take(loader, 20), rows[step:step+20])

    def test_legacy_repairs_require_correct_explicit_sampler_epoch(self):
        rows, states, _ = self.records[False]
        for step, state in states.items():
            with self.subTest(step=step):
                loader = make(True, (step - 1) // 97)
                loader.load_state_dict(state)
                self.assertEqual(take(loader, 20), rows[step:step+20])

    def test_legacy_missing_epoch_is_rejected(self):
        loader = make(True)
        with self.assertRaisesRegex(ValueError, 'validated local sampler epoch'):
            loader.load_state_dict(self.records[False][1][198])

    def test_fully_exhausted_iterator_checkpoint(self):
        for fixed in [False, True]:
            with self.subTest(fixed=fixed):
                loader = make(True, None if fixed else 1)
                loader.load_state_dict(self.records[fixed][2])
                self.assertEqual(take(loader, 20), self.records[False][0][194:214])

    def test_repeated_save_and_restore(self):
        loader = make(True)
        loader.load_state_dict(self.records[True][1][150])
        self.assertEqual(take(loader, 11), self.records[False][0][150:161])
        payload = io.BytesIO()
        torch.save(loader.state_dict(), payload)
        payload.seek(0)
        state = torch.load(payload, map_location='cpu', weights_only=False)
        resumed = make(True)
        resumed.load_state_dict(state)
        self.assertEqual(take(resumed, 20), self.records[False][0][161:181])

    def test_changed_dataset_is_rejected(self):
        loader = make(True, length=15611)
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            loader.load_state_dict(self.records[True][1][150])

    def test_corrupt_iterator_rng_is_rejected(self):
        state = deepcopy(self.records[True][1][150])
        current = state['_snapshot']['_main_snapshot']['_sampler_iter_state']['sampler_iter_state']['current']
        current[100] ^= 1
        loader = make(True)
        loader.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, 'generator differs'):
            take(loader, 1)


if __name__ == '__main__':
    unittest.main()
