import json
import unittest
from lossless_nd_candidate import LosslessNDCandidate


class CandidateContractTests(unittest.TestCase):
    def test_scalar_zero_edge_and_empty_domain(self):
        for rank,source in ((0,((),{():2**75})),(2,((0,5),{})),(3,((2,0,3),{}))):
            with self.subTest(source=source):
                model=LosslessNDCandidate(rank);model.learn([source],budget=3)
                clone=LosslessNDCandidate.from_dictionary(json.loads(json.dumps(model.dictionary())))
                self.assertEqual(clone.decode(json.loads(json.dumps(model.encode(*source)))),source)

    def test_large_negative_unseen_literals_and_odd_shape(self):
        model=LosslessNDCandidate(2)
        model.learn([((2,3),{(y,x):-3 for y in range(2) for x in range(3)})],budget=3)
        source=((3,5),{(0,4):2**80,(2,0):-2**90,(1,3):-3})
        before=json.dumps(model.dictionary(),sort_keys=True)
        packet=json.loads(json.dumps(model.encode(*source)))
        clone=LosslessNDCandidate.from_dictionary(json.loads(before))
        self.assertEqual(clone.decode(packet),source)
        self.assertEqual(before,json.dumps(model.dictionary(),sort_keys=True))

    def test_same_sequence_counterexample_is_distinguished(self):
        a=((2,2),{(0,0):1,(0,1):2,(1,0):3,(1,1):4})
        b=((2,2),{(0,0):1,(0,1):3,(1,0):2,(1,1):4})
        model=LosslessNDCandidate(2);model.learn([a,b,a,b],budget=1)
        self.assertNotEqual(model.encode(*a),model.encode(*b))

    def test_empty_domain_and_invalid_packets(self):
        model=LosslessNDCandidate(3);model.learn([((1,2,3),{})])
        packet=model.encode((3,1,5),{})
        self.assertEqual(model.decode(packet),((3,1,5),{}))
        packet['tokens']=[999]
        with self.assertRaises(ValueError):model.decode(packet)

    def test_actual_savings_counts_overlaps(self):
        cells={(0,x):1 for x in range(6)}
        for x in (8,11,14,17):cells[(0,x)]=2;cells[(0,x+1)]=3
        sample=((1,19),cells)
        a=LosslessNDCandidate(2);b=LosslessNDCandidate(2)
        la=a.learn([sample],budget=1,score='frequency')
        lb=b.learn([sample],budget=1,score='realizable')
        self.assertEqual(la,[11]);self.assertEqual(lb,[10])
        self.assertEqual(a.decode(a.encode(*sample)),sample)
        self.assertEqual(b.decode(b.encode(*sample)),sample)
