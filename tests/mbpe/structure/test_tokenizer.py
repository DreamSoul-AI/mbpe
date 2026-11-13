"""Tokenizer contracts on tiny synthetic CPU tensors. No data or training runtime."""

import os
import unittest

os.environ.setdefault("TQDM_DISABLE", "1")

import torch
from torch.utils.data import TensorDataset

from mbpe.structure.bpe.state import State
from mbpe.structure.bpe.tokenizer import Tokenizer


def _snapshot(tokenizer):
    return (
        dict(tokenizer.vocab.symbol2codeword),
        tokenizer.freq_counter.total_count,
        dict(tokenizer.freq_counter.count_table),
    )


def _tokenizer(merge=0.001, root=0.01, counter=0.01):
    return Tokenizer(
        {"root": root, "merge": merge, "freq_counter": counter},
        max_codeword_size=(1, 2, 2),
        batch_size=2,
        max_workers=0,
    )


def _train(tokenizer, tensor):
    tokenizer.train(TensorDataset(tensor), 0)


class MergeSemanticsTests(unittest.TestCase):
    def test_unknown_maximum_stops_before_a_later_known_pair(self):
        tokenizer = _tokenizer()
        tokenizer.vocab.add(("a", "b"), "M", (1, 2, 2))
        tokenizer.freq_counter.total_count = 10
        tokenizer.freq_counter.count_table = {("c", "d"): 6, ("a", "b"): 1}
        before = _snapshot(tokenizer)
        state = State(torch.zeros(1, 1, 2, 2, dtype=torch.uint8))
        state.joined = ["c", "d", "a", "b"]
        tokenizer.merge_pair(0, state, (1, 2, 2), update=False)
        self.assertEqual(state.joined, ["c", "d", "a", "b"])
        self.assertEqual(_snapshot(tokenizer), before)

    def test_overlap_is_consumed_from_the_left(self):
        tokenizer = _tokenizer()
        tokenizer.vocab.add(("a", "a"), "M", (1, 2, 2))
        tokenizer.freq_counter.total_count = 2
        tokenizer.freq_counter.count_table = {("a", "a"): 2}
        state = State(torch.zeros(1, 1, 2, 2, dtype=torch.uint8))
        state.joined = ["a", "a", "a"]
        tokenizer.merge_pair(0, state, (1, 2, 2), update=False)
        self.assertEqual(state.joined, ["M", "a"])

    def test_repeated_maximum_merges_nonoverlapping_occurrences(self):
        tokenizer = _tokenizer()
        tokenizer.vocab.add(("a", "b"), "M", (1, 2, 2))
        tokenizer.freq_counter.total_count = 3
        tokenizer.freq_counter.count_table = {("a", "b"): 2}
        state = State(torch.zeros(1, 1, 2, 2, dtype=torch.uint8))
        state.joined = ["a", "b", "a", "b"]
        tokenizer.merge_pair(0, state, (1, 2, 2), update=False)
        self.assertEqual(state.joined, ["M", "M"])


class TokenizerRoundTripTests(unittest.TestCase):
    def test_untrained_integer_samples_round_trip_without_vocab_growth(self):
        tokenizer = _tokenizer()
        before = _snapshot(tokenizer)
        single = torch.arange(16, dtype=torch.uint8).reshape(1, 1, 4, 4)
        other = torch.tensor([[[[9, 8, 7, 6], [5, 4, 3, 2], [1, 0, 12, 13], [14, 15, 11, 10]]]], dtype=torch.uint8)
        signed = torch.tensor([[[[-1, 300, 7, 8], [9, 10, 11, 12]]]], dtype=torch.int16)
        batch = torch.cat([single, other], dim=0)

        encoded_single = tokenizer.encode(single)
        self.assertEqual(encoded_single.shape, (1, 1, 4, 4))
        self.assertEqual(encoded_single.dtype, torch.uint8)
        self.assertEqual(len(encoded_single.sequences), 1)
        self.assertEqual(
            encoded_single.sequences[0],
            ("0", "2", "8", "10", "1", "3", "9", "11", "4", "6", "12", "14", "5", "7", "13", "15"),
        )
        self.assertTrue(torch.equal(tokenizer.decode(encoded_single), single))

        encoded_batch = tokenizer.encode(batch)
        self.assertEqual(len(encoded_batch.sequences), 2)
        self.assertEqual(encoded_batch.sequences[0], tokenizer.encode(single).sequences[0])
        self.assertEqual(encoded_batch.sequences[1], tokenizer.encode(other).sequences[0])
        self.assertTrue(torch.equal(tokenizer.decode(encoded_batch), batch))
        self.assertEqual(tokenizer.encode(batch), encoded_batch)

        encoded_signed = tokenizer.encode(signed)
        restored = tokenizer.decode(encoded_signed)
        self.assertEqual(restored.dtype, torch.int16)
        self.assertEqual(tuple(restored.shape), tuple(signed.shape))
        self.assertTrue(torch.equal(restored, signed))
        self.assertEqual(_snapshot(tokenizer), before)

    def test_partial_merge_round_trip_keeps_frequency_order(self):
        tokenizer = _tokenizer(merge=0.5, root=0.01, counter=0.01)
        single = torch.arange(16, dtype=torch.uint8).reshape(1, 1, 4, 4)
        _train(tokenizer, single)
        before = _snapshot(tokenizer)
        encoded = tokenizer.encode(single)
        self.assertEqual(len(encoded.sequences[0]), 4)
        self.assertEqual(tokenizer.encode(single), encoded)
        self.assertTrue(torch.equal(tokenizer.decode(encoded), single))
        self.assertEqual(_snapshot(tokenizer), before)

    def test_complete_merge_and_batch_round_trip(self):
        repeated = torch.tensor([[[[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]]], dtype=torch.uint8)
        alone = _tokenizer(merge=0.0, root=0.0, counter=0.0)
        _train(alone, repeated)
        collapsed = alone.encode(repeated)
        self.assertEqual(len(collapsed.sequences[0]), 1)
        self.assertTrue(torch.equal(alone.decode(collapsed), repeated))

        tokenizer = _tokenizer(merge=0.0, root=0.0, counter=0.0)
        single = torch.arange(16, dtype=torch.uint8).reshape(1, 1, 4, 4)
        batch = torch.cat([single, repeated], dim=0)
        _train(tokenizer, batch)
        self.assertGreater(len(tokenizer.vocab), 256)
        before = _snapshot(tokenizer)
        encoded_batch = tokenizer.encode(batch)
        self.assertEqual(tokenizer.encode(single).sequences[0], encoded_batch.sequences[0])
        self.assertEqual(tokenizer.encode(repeated).sequences[0], encoded_batch.sequences[1])
        self.assertTrue(torch.equal(tokenizer.decode(encoded_batch), batch))
        self.assertTrue(torch.equal(tokenizer.decode(tokenizer.encode(single)), single))
        self.assertEqual(tokenizer.encode(batch), encoded_batch)
        self.assertEqual(_snapshot(tokenizer), before)


class TokenizerInputTests(unittest.TestCase):
    def test_illegal_tensors_and_encodings_fail(self):
        tokenizer = _tokenizer()
        valid = torch.arange(16, dtype=torch.uint8).reshape(1, 1, 4, 4)
        with self.assertRaises(TypeError):
            tokenizer.encode(None)
        with self.assertRaises(TypeError):
            tokenizer.decode(["0", "1"])
        for illegal in (
            torch.zeros(1, 1, 2, 2, dtype=torch.float32),
            torch.zeros(1, 1, 2, 2, dtype=torch.bool),
            torch.zeros(0, 1, 2, 2, dtype=torch.uint8),
            torch.arange(4, dtype=torch.uint8).reshape(1, 2, 2),
            torch.arange(12, dtype=torch.uint8).reshape(1, 1, 3, 4),
        ):
            with self.subTest(shape=tuple(illegal.shape), dtype=illegal.dtype):
                with self.assertRaises(ValueError):
                    tokenizer.encode(illegal)
        if torch.cuda.is_available():
            with self.assertRaises(ValueError):
                tokenizer.encode(valid.to("cuda"))
        encoded = tokenizer.encode(valid)
        missing = type(encoded)((("missing-code",),), encoded.shape, encoded.dtype)
        with self.assertRaises(ValueError):
            tokenizer.decode(missing)
