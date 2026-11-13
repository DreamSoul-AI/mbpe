import unittest
from elsner_rule_decode import rule_shapes,decode_sequence


class OfficialRuleDecodeTests(unittest.TestCase):
    def test_nonconvex_shape_keeps_negative_column_offset(self):
        shapes=rule_shapes([(1,2,-1,1)])
        self.assertEqual(shapes[256],{(0,0):1,(1,-1):2})
        grid=decode_sequence([(3,0,0),(256,0,1),(4,1,1)],shapes,(2,2))
        self.assertEqual(grid,[[3,1],[2,4]])

    def test_rule_decoder_rejects_missing_or_duplicate_coverage(self):
        shapes=rule_shapes([])
        with self.assertRaises(ValueError):decode_sequence([(1,0,0)],shapes,(2,2))
        with self.assertRaises(ValueError):decode_sequence([(1,0,0),(2,0,0)],shapes,(1,1))
