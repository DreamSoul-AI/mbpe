"""A symbol cap stops new vocabulary entries. Omitting it keeps the old training."""

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


def _grid():
    return torch.tensor(
        [[[[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]]],
        dtype=torch.uint8,
    )


def _train(max_symbols=None):
    tokenizer = Tokenizer(
        {"root": 0.0, "merge": 0.0, "freq_counter": 0.0},
        max_codeword_size=(1, 2, 2),
        batch_size=2,
        max_workers=0,
        max_symbols=max_symbols,
    )
    tokenizer.train(TensorDataset(_grid()), 0)
    return tokenizer


class SymbolCapTests(unittest.TestCase):
    def test_cap_stops_and_missing_field_is_unlimited(self):
        open_ended = _train()
        same = _train()
        capped = _train(257)
        self.assertEqual(len(open_ended.vocab), len(same.vocab))
        self.assertEqual(open_ended.merge_order, same.merge_order)
        self.assertGreater(len(open_ended.vocab), 257)
        self.assertEqual(len(capped.vocab), 257)
        self.assertLess(len(capped.vocab), len(open_ended.vocab))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            save_tokenizer(path, open_ended)
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(document["config"]["max_symbols"])
            del document["config"]["max_symbols"]
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = load_tokenizer(path)
        self.assertIsNone(loaded.max_symbols)
        self.assertEqual(len(loaded.vocab), len(open_ended.vocab))
