"""Synthetic grids are fixed by the manifest version and the seed."""

import unittest

from mbpe.structure.synthetic import generate_grids, load_manifest, select_grids


class SyntheticManifestTests(unittest.TestCase):
    def test_same_seed_matches_and_other_seed_differs(self):
        first = generate_grids(0)
        second = generate_grids(0)
        other = generate_grids(1)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual([grid["shape"] for grid in first], [(1, 2, 2), (1, 2, 4), (1, 4, 4)])
        self.assertEqual(load_manifest(1)["manifest_version"], 1)

    def test_version_two_is_deterministic_and_mixed(self):
        first = generate_grids(0, version=2)
        self.assertEqual(first, generate_grids(0, version=2))
        self.assertNotEqual(first, generate_grids(1, version=2))
        self.assertEqual([grid["shape"] for grid in first], [(1, 16, 16), (1, 32, 32), (1, 24, 40)])
        stripe, tile, noise = first
        width = 16
        row = stripe["values"][:width]
        self.assertEqual(stripe["values"][: width * 16], row * 16)
        self.assertEqual(len(set(row)), 4)
        self.assertEqual(tile["values"][0], tile["values"][4])
        self.assertEqual(tile["values"][0], tile["values"][32 * 4])
        self.assertGreater(len(set(noise["values"])), 16)
        with self.assertRaises(ValueError):
            generate_grids(0, version=3)
        tile = select_grids(first, "tile")
        self.assertEqual([grid["name"] for grid in tile], ["tile"])
        self.assertEqual(tile[0]["shape"], (1, 32, 32))
        self.assertEqual(select_grids(first, "stripe")[0]["shape"], (1, 16, 16))
        self.assertEqual(select_grids(first, "noise")[0]["shape"], (1, 24, 40))
        self.assertEqual(select_grids(first), first)
        with self.assertRaises(ValueError):
            select_grids(first, "missing")
