"""Offline contracts for import isolation and configuration preview."""

import json
import subprocess
import sys
import unittest

from mbpe.structure.config import TokenizerConfig


class SetupTests(unittest.TestCase):
    def test_invalid_external_configuration(self):
        for kwargs in ({"batch_size": 0}, {"batch_size": True},
                       {"max_codeword_size": (1, 0)}, {"root_min_freq": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TokenizerConfig(**kwargs)

    def test_preview_does_not_import_tensor_runtime(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; from mbpe.flow.cli import main; main(['plan']); assert 'torch' not in sys.modules"],
            check=True, text=True, capture_output=True)
        self.assertEqual(json.loads(result.stdout)["tokenizer_kwargs"]["min_freq"]["merge"], 0.001)

    def test_invalid_cli_fails_without_a_plan(self):
        result = subprocess.run([sys.executable, "-m", "mbpe", "plan", "--patch-size", "0"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
