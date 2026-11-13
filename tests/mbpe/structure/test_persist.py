"""Tokenizer version-1 JSON round trip. Temporary files are removed with the test."""

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import torch
from torch.utils.data import TensorDataset

from mbpe.structure.bpe.persist import load_tokenizer, save_tokenizer
from mbpe.structure.bpe.tokenizer import Tokenizer


def _tokenizer():
    return Tokenizer(
        {"root": 0.0, "merge": 0.0, "freq_counter": 0.0},
        max_codeword_size=(1, 2, 2),
        batch_size=2,
        max_workers=0,
    )


class TokenizerPersistTests(unittest.TestCase):
    def test_saved_tokenizer_reloads_merges_config_and_encoding(self):
        tokenizer = _tokenizer()
        grid = torch.tensor(
            [[[[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]]],
            dtype=torch.uint8,
        )
        tokenizer.train(TensorDataset(grid), 0)
        self.assertGreater(len(tokenizer.merge_order), 0)
        before = tokenizer.encode(grid)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            save_tokenizer(path, tokenizer)
            loaded = load_tokenizer(path)

        self.assertEqual(loaded.merge_order, tokenizer.merge_order)
        self.assertEqual(loaded.batch_size, tokenizer.batch_size)
        self.assertEqual(loaded.max_workers, tokenizer.max_workers)
        self.assertEqual(tuple(loaded.codeword_sizes[0]), (1, 2, 2))
        self.assertEqual(loaded.freq_counter.min_freq, tokenizer.freq_counter.min_freq)
        self.assertEqual(loaded.freq_counter.total_count, tokenizer.freq_counter.total_count)
        encoded = loaded.encode(grid)
        self.assertEqual(encoded, before)
        self.assertEqual(encoded, tokenizer.encode(grid))
        self.assertTrue(torch.equal(loaded.decode(encoded), grid))

    def test_unknown_version_fails(self):
        tokenizer = _tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            save_tokenizer(path, tokenizer)
            document = json.loads(path.read_text(encoding="utf-8"))
            document["version"] = 2
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_tokenizer(path)
        self.assertIn("2", str(caught.exception))
