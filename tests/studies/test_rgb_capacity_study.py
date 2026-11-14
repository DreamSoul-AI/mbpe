import unittest
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from rgb_capacity_study import balanced_subset, select_indices


class RGBCapacitySelectionTests(unittest.TestCase):
    def test_selection_module_does_not_require_svhn_reader_dependency(self):
        environment=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]/'studies'))
        result=subprocess.run([sys.executable,'-B','-c',
            "import sys; sys.modules['scipy']=None; "
            "from rgb_capacity_study import select_indices; assert callable(select_indices)"],
            env=environment,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_selection_is_balanced_deterministic_and_excludes_prior_inputs(self):
        labels = np.repeat(np.arange(10), 5)
        images = np.arange(50, dtype=np.uint8).reshape(50, 1, 1, 1)
        first = select_indices(labels, images, 20, 111401, {0, 5}, set())
        second = select_indices(labels, images, 20, 111401, {0, 5}, set())
        self.assertEqual(first, second)
        self.assertFalse({0, 5} & {row['index'] for row in first})
        self.assertEqual([sum(row['class'] == k for row in first) for k in range(10)], [2]*10)
        seen = {row['sha256'] for row in first}
        independent = select_indices(labels, images, 20, 111402, {0, 5}, seen)
        self.assertFalse({row['sha256'] for row in first} & {row['sha256'] for row in independent})
        subset = balanced_subset(first, 10)
        self.assertEqual([first[i]['class'] for i in subset], list(range(10)))

    def test_insufficient_unique_data_fails_instead_of_changing_protocol(self):
        labels = np.arange(10)
        for count, images in ((20, np.arange(10, dtype=np.uint8).reshape(10, 1)),
                              (10, np.zeros((10, 1), dtype=np.uint8))):
            with self.subTest(count=count), self.assertRaises(ValueError):
                select_indices(labels, images, count, 1, set(), set())
        with self.assertRaises(ValueError):
            balanced_subset([{'class': k} for k in range(10)], 11)


if __name__ == '__main__':
    unittest.main()
