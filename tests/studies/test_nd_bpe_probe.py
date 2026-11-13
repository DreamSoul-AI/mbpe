"""Representation and boundary-graph regressions, no data/model access."""
import json
import unittest

from nd_bpe_probe import AxisBPE, State, sample
from nd_bpe_graph_probe import GraphState
from nd_bpe_incremental_probe import Index, IndexedState
from collections import Counter


class NDProbeTests(unittest.TestCase):
    def test_incremental_counts_cover_overlapping_repeated_pairs_and_zero_removal(self):
        from itertools import product
        for shape in ((4,),(2,4),(3,2,2),(2,2,2,2),(2,)*8):
            with self.subTest(shape=shape):
                points=list(product(*(range(n) for n in shape)))
                samples=[(shape,{p:value for p in points}) for value in (1,2)]
                self._check_incremental(samples)

    def _check_incremental(self,samples):
        reference=[State(*s) for s in samples]
        index=Index()
        indexed=[IndexedState(*s,index,i) for i,s in enumerate(samples)]
        for step in range(20):
            pairs=[s.pairs() for s in reference]
            counts=Counter()
            for row in pairs:counts.update({k:len(v) for k,v in row.items()})
            self.assertEqual(counts,index.counts)
            key,frequency=index.best()
            if not counts:
                self.assertEqual((key,frequency),(None,0));break
            expected=min(counts,key=lambda k:(-counts[k],k))
            self.assertEqual(key,expected)
            candidates=[]
            for image,a,b in index.occurrences[key]:
                state=indexed[image]
                if state.regions[a][1]>state.regions[b][1]:a,b=b,a
                candidates.append((state.edges[a][b],image,a,b))
            for state,row in zip(reference,pairs):state.merge(row.get(key,[]),256+step)
            for _,image,a,b in sorted(candidates):indexed[image].replace(a,b,256+step)
            self.assertEqual([s.regions for s in reference],[s.regions for s in indexed])

    def test_mixed_training_shapes_allow_unseen_test_shape(self):
        model=AxisBPE(2)
        training=[sample(shape,p,True) for p,shape in enumerate(((1,5),(3,2),(2,7)))]
        model.learn(training,budget=8)
        source=sample((7,3),3,True)
        encoded=model.encode(*source)
        self.assertEqual(model.decode(encoded),source)
        self.assertEqual(model.decode_without_anchors(model.without_anchors(encoded,source[1])),source)

    def test_sparse_dictionary_and_packet_survive_json_roundtrip(self):
        shape=(3,4,3)
        model=AxisBPE(3)
        model.learn([sample(shape,p,True) for p in range(3)],budget=10)
        source=sample(shape,3,True)
        before=json.dumps(model.dictionary(),sort_keys=True)
        encoded=model.encode(*source)
        compact=model.without_anchors(encoded,source[1])
        clone=AxisBPE.from_dictionary(json.loads(before))
        self.assertEqual(clone.decode(json.loads(json.dumps(encoded))),source)
        self.assertEqual(clone.decode_without_anchors(json.loads(json.dumps(compact))),source)
        self.assertEqual(before,json.dumps(model.dictionary(),sort_keys=True))

    def test_incomplete_or_overlapping_packet_is_rejected(self):
        model=AxisBPE(2)
        source=sample((2,3))
        encoded=model.encode(*source)
        encoded['tokens'].append(encoded['tokens'][0])
        with self.assertRaises(ValueError):model.decode(encoded)
        compact=model.without_anchors(model.encode(*source),source[1])
        compact['tokens'].pop()
        with self.assertRaises(ValueError):model.decode_without_anchors(compact)

    def test_boundary_graph_preserves_shared_boundary_order(self):
        source=sample((5,6),holes=True)
        reference=State(*source);graph=GraphState(*source)
        for step in range(12):
            a=reference.pairs();b=graph.pairs()
            self.assertEqual(a,b)
            if not a:break
            key=min(a,key=lambda k:(-len(a[k]),k))
            reference.merge(a[key],256+step);graph.merge(b[key],256+step)
            self.assertEqual(reference.regions,graph.regions)

    def test_dimension_rank_and_absent_cells_are_not_silent_padding(self):
        source=((2,2),{(0,0):0,(1,1):255})
        model=AxisBPE(2)
        encoded=model.encode(*source)
        compact=model.without_anchors(encoded,source[1])
        self.assertEqual(model.decode_without_anchors(compact),source)
        with self.assertRaises(ValueError):model.encode((4,),{(0,):0})
