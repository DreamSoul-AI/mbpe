"""CompressionMetrics must read a vocabulary map that actually exists."""

import os
import unittest

os.environ.setdefault("TQDM_DISABLE", "1")

import torch  # 先于 mbpe 加载，避免本机单独导入 numpy 时的动态库失败

from mbpe.structure.bpe.compression_metrics import CompressionMetrics
from mbpe.structure.bpe.tokenizer import Tokenizer


class CompressionMetricsTests(unittest.TestCase):
    def test_entropy_diversity_uses_existing_codeword_map(self):
        tokenizer = Tokenizer(
            {"root": 0.01, "merge": 0.001, "freq_counter": 0.01},
            max_codeword_size=(1, 2, 2),
            batch_size=4,
            max_workers=0,
        )
        metrics = CompressionMetrics(tokenizer)
        result = metrics.compute_entropy_based_compression(["0", "0", "1"])
        self.assertEqual(result["unique_tokens"], 2)
        self.assertEqual(result["token_diversity"], 2 / len(tokenizer.vocab.codeword2symbol))
        self.assertFalse(hasattr(tokenizer.vocab, "codewords"))
        self.assertGreater(result["entropy"], 0)
