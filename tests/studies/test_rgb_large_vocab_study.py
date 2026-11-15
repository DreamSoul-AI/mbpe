import unittest
import numpy as np
from rgb_large_vocab_study import texts,expand
import rgb_input_codec as codec


class LargeVocabularyGridTests(unittest.TestCase):
    def test_column_atoms_and_residual_preserve_rgb_extremes(self):
        image=np.zeros((3,32,32),dtype=np.uint8)
        image[0,1,0]=255;image[0,0,1]=127;image[1,31,31]=128;image[2,0,0]=254
        strings=texts(image)
        self.assertEqual(ord(strings[0][1])-256,255)
        self.assertEqual(ord(strings[0][32])-256,127)
        residual=codec.difference(image)
        decoded=np.stack([expand([3],{3:string}) for string in texts(residual)])
        np.testing.assert_array_equal(codec.restore_difference(decoded),image)

    def test_invalid_expansion_is_rejected(self):
        for text in (chr(256)*1023,chr(256)*1025,chr(512)*1024):
            with self.assertRaises(ValueError):expand([0],{0:text})


if __name__=='__main__':unittest.main()
