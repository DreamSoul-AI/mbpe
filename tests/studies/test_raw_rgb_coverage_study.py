import importlib.util
import unittest
from hashlib import sha256

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('tokenizers'), 'optional tokenizers research runtime')
class RawRGBStreamTests(unittest.TestCase):
    def test_spatial_and_channel_difference_orders(self):
        from raw_rgb_coverage_study import input_strings, restore_stream
        from rgb_large_vocab_study import expand

        words = {i: chr(256 + i) for i in range(256)}
        image = np.random.default_rng(27).integers(0, 256, (3, 32, 32), dtype=np.uint8)
        rows = [{'sha256': sha256(image.tobytes()).hexdigest()}]
        for basis in ('raw', 'color', 'chain1', 'chain2'):
            for order in range(4):
                with self.subTest(basis=basis, order=order):
                    strings = input_strings(image, basis, order=order)
                    ids = [ord(value) - 256 for channel in strings for value in channel]
                    restore_stream(ids, words, rows, basis, order=order)
        # The third horizontal difference of a quadratic ramp is zero after the boundary.
        ramp = np.broadcast_to(np.arange(32) ** 2, (3, 32, 32)).astype(np.uint8)
        for string in input_strings(ramp, 'raw', order=3):
            grid = expand([ord(value) - 256 for value in string], words)
            self.assertTrue(np.all(grid[:, 3:] == 0))
        # An RGB arithmetic progression has zero second channel difference.
        color_ramp = np.broadcast_to(np.array([250, 2, 10], dtype=np.uint8)[:, None, None], (3, 32, 32))
        result = input_strings(color_ramp, 'chain2', order=0)
        self.assertEqual(result[2], chr(256) * 1024)
        for direction, order in [('vertical', 1), ('mixed', 2)]:
            strings = input_strings(image, 'color', order=order, direction=direction)
            ids = [ord(value) - 256 for channel in strings for value in channel]
            restore_stream(ids, words, rows, 'color', order=order, direction=direction)

    def test_frozen_transforms_use_same_raw_identity_for_decode(self):
        from raw_rgb_coverage_study import input_strings, restore_stream

        image = np.random.default_rng(17).integers(0, 256, (3, 32, 32), dtype=np.uint8)
        words = {i: chr(256 + i) for i in range(256)}
        rows = [{'sha256': sha256(image.tobytes()).hexdigest()}]
        for transform in ('raw', 'difference', 'color'):
            with self.subTest(transform=transform):
                stream = [ord(c) - 256 for channel in input_strings(image, transform) for c in channel]
                restore_stream(stream, words, rows, transform)
                if transform != 'raw':
                    with self.assertRaisesRegex(ValueError, 'independent decode differs'):
                        restore_stream(stream, words, rows)
        for scan in ('row', 'hilbert'):
            with self.subTest(scan=scan):
                stream = [ord(c) - 256 for channel in input_strings(image, 'color', scan) for c in channel]
                restore_stream(stream, words, rows, 'color', scan)
                with self.assertRaisesRegex(ValueError, 'independent decode differs'):
                    restore_stream(stream, words, rows, 'color')

    def test_decode_reconstructs_boundaries_without_counts(self):
        from raw_rgb_coverage_study import restore_stream

        words = {0: chr(256) * 2, 1: chr(257) * 4}
        first = np.zeros((3, 32, 32), dtype=np.uint8)
        second = np.ones_like(first)
        rows = [{'sha256': sha256(image.tobytes()).hexdigest()} for image in (first, second)]
        ids = [0] * (3 * 512) + [1] * (3 * 256)
        restore_stream(ids, words, rows)
        for damaged in (ids[:-1], ids + [0], [0] * 511 + [1] + ids[512:]):
            with self.assertRaises(ValueError):
                restore_stream(damaged, words, rows)

    def test_uniform_cost_uses_fewer_words_than_ranked_merge(self):
        from tokenizers import Tokenizer, models

        vocab = {word: i for i, word in enumerate(('A', 'B', 'C', 'D', 'BC', 'AB', 'CD'))}
        bpe = Tokenizer(models.BPE(vocab=vocab, merges=[('B', 'C'), ('A', 'B'), ('C', 'D')]))
        minimum = Tokenizer(models.Unigram([(word, -1.0) for word in vocab]))
        self.assertEqual(minimum.get_vocab(), bpe.get_vocab())
        self.assertEqual(bpe.encode('ABCD').tokens, ['A', 'BC', 'D'])
        self.assertEqual(minimum.encode('ABCD').tokens, ['AB', 'CD'])
