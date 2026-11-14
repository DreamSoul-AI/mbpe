import unittest

import numpy as np
import torch

import rgb_input_codec as codec
from rgb_linear_codec import LinearRGBGrammar,SCANS,learn_rgb


class RGBLinearCodecTests(unittest.TestCase):
    def test_all_scans_preserve_pixels_and_inband_channel_counts(self):
        image=np.arange(3072,dtype=np.uint16).astype(np.uint8).reshape(3,32,32)
        for scan in SCANS:
            model=LinearRGBGrammar.from_dictionary(LinearRGBGrammar(scan,[(0,1)]).dictionary())
            for residual in (False,True):
                transformed=codec.difference(image) if residual else image
                blob,_=codec.encode_bpe(transformed,model)
                tensor=codec.to_bf16(blob,192,torch)
                restored=codec.restore_input(tensor,192,image.shape,model,residual=residual,bpe=True,torch=torch)
                self.assertTrue(np.array_equal(restored,image))

    def test_learning_uses_existing_budget_and_rejects_malformed_rules(self):
        images=np.zeros((2,3,32,32),dtype=np.uint8)
        images[:,:,:,1::2]=1
        model=learn_rgb(images,'row',4)
        self.assertEqual(len(model.rules),4)
        packet=model.encode((32,32),{p:int(v) for p,v in np.ndenumerate(images[0,0])})
        self.assertEqual(model.decode(packet)[1],{p:int(v) for p,v in np.ndenumerate(images[0,0])})
        for rules in ([(256,0)],[(0,-1)],[(0,True)]):
            with self.assertRaises(ValueError):LinearRGBGrammar('row',rules)
        packet['tokens']=[0]
        with self.assertRaises(ValueError):model.decode(packet)


if __name__=='__main__':unittest.main()
