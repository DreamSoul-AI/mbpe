import unittest
import numpy as np
from zhang_direction_trace import decode_with_trace,pack_directions,unpack_directions


class DirectionTraceTests(unittest.TestCase):
    def test_one_bit_distinguishes_original_collision(self):
        rules=[((1,2),256)]
        a=decode_with_trace([256,3,4],(2,2),rules,[0])
        b=decode_with_trace([256,3,4],(2,2),rules,[1])
        self.assertEqual(a.tolist(),[[1,2],[3,4]])
        self.assertEqual(b.tolist(),[[1,3],[2,4]])
        self.assertFalse(np.array_equal(a,b))

    def test_direction_payload_and_missing_trace(self):
        directions=[1,0,1,1,0,0,0,1,1]
        self.assertEqual(unpack_directions(pack_directions(directions),len(directions)),directions)
        with self.assertRaises(ValueError):decode_with_trace([256,3,4],(2,2),[((1,2),256)],[])
