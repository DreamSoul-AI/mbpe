import unittest
import numpy as np
from color_template_codec import TemplateCodec,color_basis,inverse_color_basis,patch_key,fit_templates


class ColorTemplateTests(unittest.TestCase):
    def test_color_basis_extremes_and_random_are_exact(self):
        image=np.random.default_rng(19).integers(0,256,(3,32,32),dtype=np.uint8)
        image[:,0,0]=[0,255,128]
        np.testing.assert_array_equal(inverse_color_basis(color_basis(image)),image)
        with self.assertRaises(ValueError):color_basis(image.astype(np.int16))

    def test_shifted_pattern_reuses_template_but_pays_anchor(self):
        pattern=np.arange(64,dtype=np.uint8).reshape(8,8)
        model=TemplateCodec(True,[(8,patch_key(pattern,True))])
        image=np.stack([np.tile(np.add(pattern,shift,dtype=np.uint8),(4,4)) for shift in (17,128,253)])
        tokens,stats=model.encode(image)
        self.assertEqual(len(tokens),3*16*2)
        self.assertEqual(stats['templates_8'],48)
        np.testing.assert_array_equal(model.decode(tokens),image)
        restored=TemplateCodec.from_dictionary(model.dictionary())
        np.testing.assert_array_equal(restored.decode(tokens),image)
        absolute=TemplateCodec(False,[(8,pattern.tobytes())])
        self.assertGreater(len(absolute.encode(image)[0]),len(tokens))

    def test_noise_and_mixed_sizes_preserve_implicit_anchor_order(self):
        image=np.random.default_rng(44).integers(0,256,(3,32,32),dtype=np.uint8)
        model=TemplateCodec(False,[])
        ids,_=model.encode(image);self.assertEqual(len(ids),3072)
        np.testing.assert_array_equal(model.decode(ids),image)
        entries=[(s,image[0,:s,:s].tobytes()) for s in (2,4,8)]
        model=TemplateCodec(False,entries)
        ids,_=model.encode(image);self.assertLess(len(ids),3072)
        np.testing.assert_array_equal(model.decode(ids),image)
        fitted,_=fit_templates([image],True,8)
        ids,_=fitted.encode(image)
        np.testing.assert_array_equal(fitted.decode(ids),image)

    def test_bad_dictionary_or_payload_fails(self):
        for entries in ([(3,b'\0'*9)],[(2,b'\1'*4)],[(2,b'\0'*4)]*2):
            with self.assertRaises(ValueError):TemplateCodec(True,entries)
        model=TemplateCodec(True,[(8,b'\0'*64)])
        for ids in ([256],[256,256],[257],[-1],[],[0]*3073,[0]*31+[256,0]):
            with self.assertRaises(ValueError):model.decode(ids)


if __name__=='__main__':unittest.main()
