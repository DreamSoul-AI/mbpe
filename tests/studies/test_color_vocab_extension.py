import unittest
from unittest.mock import patch
import numpy as np
import color_vocab_extension as extension
from rgb_large_vocab_study import texts,expand


class RepresentationTests(unittest.TestCase):
    def test_raw_bypasses_all_transforms_and_preserves_column_atoms(self):
        image=np.random.default_rng(17).integers(0,256,(3,32,32),dtype=np.uint8)
        with patch.object(extension,'difference',side_effect=AssertionError('raw must not difference')),patch.object(extension,'color_basis',side_effect=AssertionError('raw must not transform colors')):
            raw=extension.prepare_image(image,'raw')
        self.assertIs(raw,image)
        words={i:chr(256+i) for i in range(256)}
        grids=[expand([ord(c)-256 for c in s],words) for s in texts(raw)]
        with patch.object(extension,'restore_difference',side_effect=AssertionError('raw must not inverse difference')),patch.object(extension,'inverse_color_basis',side_effect=AssertionError('raw must not inverse colors')):
            np.testing.assert_array_equal(extension.restore_image(grids,'raw'),image)

    def test_existing_representations_and_invalid_inputs(self):
        image=np.random.default_rng(18).integers(0,256,(3,32,32),dtype=np.uint8)
        for kind in ['raw','baseline','color']:
            np.testing.assert_array_equal(extension.restore_image(extension.prepare_image(image,kind),kind),image)
        with self.assertRaises(ValueError):extension.prepare_image(image,'unknown')
        with self.assertRaises(ValueError):extension.prepare_image(image.astype(np.int16),'raw')
