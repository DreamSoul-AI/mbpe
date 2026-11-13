"""Smoke test for the run command. Output stays in a temporary directory."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")


class RunCommandTests(unittest.TestCase):
    def test_one_seed_writes_summary_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-m", "mbpe", "run", "--seeds", "0", "--output", directory],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["seeds"], [0])
            self.assertEqual(summary["manifest_version"], 1)
            self.assertTrue(summary["mean"]["lossless"])
            self.assertIn("token_count_delta", summary["per_seed"][0])
            self.assertIn("compression_ratio", summary["mean"])
            self.assertTrue((Path(directory) / "tokenizer-seed-0.json").is_file())
