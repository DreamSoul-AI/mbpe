"""Run from any working directory using the standard library test runner."""

from pathlib import Path
import os
import sys
import unittest

root = Path(__file__).resolve().parents[1]
source_paths = [str(root / "src"), str(root / "studies")]
sys.path[:0] = source_paths
os.environ["PYTHONPATH"] = os.pathsep.join([*source_paths, os.environ.get("PYTHONPATH", "")])
suite = unittest.defaultTestLoader.discover(
    str(root / "tests"), pattern="test_*.py", top_level_dir=str(root),
)
raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
