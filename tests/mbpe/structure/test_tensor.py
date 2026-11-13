"""Small CPU checks only; no dataset access or tokenizer training."""

import unittest

import torch
from mbpe.structure.bpe.patch import Patchify, Reconstruct
from mbpe.structure.bpe.utils import compress_indices, decompress_indices


class TensorTests(unittest.TestCase):
    def test_patch_reconstruction_preserves_values(self):
        for shape in ((1, 1, 4, 4), (2, 1, 3, 8, 8)):
            with self.subTest(shape=shape):
                x = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
                patched = Patchify((1, 2, 2))(x)
                rebuilt = Reconstruct(shape[-3:])(patched)
                self.assertTrue(torch.equal(x, rebuilt))

    def test_index_serialization_roundtrip(self):
        for indices in ([], [0], [1, 2, 3, 7, 9, 10]):
            self.assertEqual(decompress_indices(compress_indices(indices)), indices)
