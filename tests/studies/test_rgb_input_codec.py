import struct
import unittest

import numpy as np
import torch

from lossless_nd_candidate import LosslessNDCandidate
import rgb_input_codec as codec


class RGBInputCodecTests(unittest.TestCase):
    def model(self):
        return LosslessNDCandidate.from_dictionary({
            'format': 'nd-bpe-candidate-dictionary', 'version': 1, 'rank': 2,
            'base': list(range(256)), 'rules': [[[0, 0, [0, 1]], 256]],
        })

    def test_rgb_residual_and_full_input_round_trip(self):
        image = np.array([[[0, 255, 0], [255, 0, 255]],
                          [[3, 3, 3], [0, 0, 0]],
                          [[254, 255, 1], [2, 2, 2]]], dtype=np.uint8)
        model = self.model()
        for residual in (False, True):
            transformed = codec.difference(image) if residual else image
            self.assertTrue(np.array_equal(codec.restore_difference(codec.difference(image)), image))
            for bpe in (False, True):
                blob = codec.encode_bpe(transformed, model)[0] if bpe else transformed.tobytes()
                for capacity in (2, 3, 192):
                    with self.subTest(residual=residual, bpe=bpe, capacity=capacity):
                        tensor = codec.to_bf16(blob, capacity, torch)
                        restored = codec.restore_input(tensor, capacity, image.shape, model,
                                                       residual=residual, bpe=bpe, torch=torch)
                        self.assertTrue(np.array_equal(restored, image))
                        self.assertEqual(tensor.numel() * tensor.element_size(), len(tensor) * 384)

    def test_rejects_invalid_payloads_and_coverage(self):
        model = self.model()
        image = np.zeros((1, 2, 3), dtype=np.uint8)
        blob, _ = codec.encode_bpe(image, model)
        for bad in (blob[:-1], blob + b'\0', b'\0\0', struct.pack('<H', 7) + b'\0' * 8,
                    struct.pack('<H', 1) + b'\x80\0'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                codec.decode_bpe(bad, model, image.shape)
        # A legal symbol count with insufficient spatial coverage must still fail.
        with self.assertRaises(ValueError):
            codec.decode_bpe(struct.pack('<H', 1) + b'\0\0', model, image.shape)

    def test_rejects_padding_fraction_and_hidden_dimensions(self):
        shape = (1, 1, 1)
        tensor = codec.to_bf16(b'\x01', 3, torch)
        for column, value in ((1, 1), (0, .5), (0, 256), (3, 1), (0, float('nan'))):
            malformed = tensor.clone()
            malformed[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                codec.from_bf16(malformed, 3, shape, bpe=False, torch=torch)
        with self.assertRaises(ValueError):
            codec.from_bf16(torch.cat([tensor, tensor * 0]), 3, shape, bpe=False, torch=torch)
        with self.assertRaises(ValueError):
            codec.from_bf16(tensor[:0], 3, shape, bpe=False, torch=torch)

    def test_uint8_contract_and_modulo_boundary(self):
        for shape in ((1, 2, 0), (2, 3, 4), (3, 256, 256)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                codec.validate_shape(shape)
        with self.assertRaises(ValueError):
            codec.difference(np.zeros((3, 2, 2), dtype=np.int32))
        for value in (0, 1, 254, 255):
            image = np.full((3, 1, 4), value, dtype=np.uint8)
            self.assertTrue(np.array_equal(codec.restore_difference(codec.difference(image)), image))


if __name__ == '__main__':
    unittest.main()
