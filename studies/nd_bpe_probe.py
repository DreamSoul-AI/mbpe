"""Small CPU research prototype: axis adjacency and explicit N-D shape/anchors.

Not an optimized codec and not a replacement for either official baseline.
"""
from collections import Counter
import base64
from itertools import product
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


class State:
    def __init__(self, shape, cells, alphabet_size=256):
        if not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('positive shape required')
        self.shape = tuple(shape)
        self.cells = {}
        self.regions = {}
        for uid, (coord, value) in enumerate(sorted(cells.items())):
            if len(coord) != len(shape) or any(not 0 <= x < n for x, n in zip(coord, shape)):
                raise ValueError('coordinate outside shape')
            if type(value) is not int or not 0 <= value < alphabet_size:
                raise ValueError('value outside explicit base alphabet')
            self.cells[coord] = uid
            self.regions[uid] = (value, coord, frozenset([coord]))
        self.next_id = len(self.regions)

    def pairs(self):
        matches = {}
        visited = set()
        for point, uid in sorted(self.cells.items()):
            for axis in range(len(self.shape)):
                neighbor = list(point); neighbor[axis] += 1
                other = self.cells.get(tuple(neighbor))
                if other is None or uid == other:
                    continue
                left, right = uid, other
                if self.regions[left][1] > self.regions[right][1]:
                    left, right = right, left
                if (left, right) in visited:
                    continue
                visited.add((left, right))
                a, anchor_a, _ = self.regions[left]
                b, anchor_b, _ = self.regions[right]
                key = (a, b, sub(anchor_b, anchor_a))
                matches.setdefault(key, []).append((left, right))
        return matches

    def merge(self, candidates, token):
        # Stable nonoverlapping acceptance; no speculative gain claims.
        count = 0
        for left, right in candidates:
            if left not in self.regions or right not in self.regions:
                continue
            _, anchor, cells_a = self.regions.pop(left)
            _, _, cells_b = self.regions.pop(right)
            cells = cells_a | cells_b
            uid = self.next_id; self.next_id += 1
            self.regions[uid] = (token, anchor, cells)
            for coord in cells:
                self.cells[coord] = uid
            count += 1
        return count


class AxisBPE:
    def __init__(self, rank):
        self.rank = rank
        zero = (0,)*rank
        self.shapes = {value: {zero: value} for value in range(256)}
        self.rules = []

    def learn(self, samples, budget=16):
        states = [State(shape, cells) for shape, cells in samples]
        for _ in range(budget):
            sets = [state.pairs() for state in states]
            counts = Counter()
            for pairs in sets:
                counts.update({key: len(matches) for key, matches in pairs.items()})
            if not counts:
                break
            key = min(counts, key=lambda k: (-counts[k], k))
            if counts[key] < 2:
                break
            a, b, delta = key
            token = 256+len(self.rules)
            atoms = dict(self.shapes[a])
            for offset, value in self.shapes[b].items():
                coord = add(offset, delta)
                if coord in atoms:
                    raise ValueError('overlapping rule shapes')
                atoms[coord] = value
            self.shapes[token] = atoms
            self.rules.append((key, token))
            for state, pairs in zip(states, sets):
                state.merge(pairs.get(key, []), token)

    def encode(self, shape, cells):
        if len(shape) != self.rank:
            raise ValueError('rank does not match dictionary')
        state = State(shape, cells)
        for key, token in self.rules:
            state.merge(state.pairs().get(key, []), token)
        return {'shape': list(shape), 'tokens': [[token, list(anchor)] for token, anchor, _ in
                sorted(state.regions.values(), key=lambda item: item[1])]}

    def decode(self, packet):
        shape = packet['shape']
        if len(shape) != self.rank:
            raise ValueError('rank does not match dictionary')
        cells = {}
        for token, anchor in packet['tokens']:
            if len(anchor) != self.rank or token not in self.shapes:
                raise ValueError('invalid token or anchor')
            for offset, value in self.shapes[token].items():
                coord = add(anchor, offset)
                if coord in cells or any(not 0 <= x < n for x, n in zip(coord, shape)):
                    raise ValueError('overlapping or out-of-bound decoded cell')
                cells[coord] = value
        return tuple(shape), cells

    def dictionary(self):
        return {'rank': self.rank, 'base_alphabet': 256, 'rules': self.rules}

    @classmethod
    def from_dictionary(cls, dictionary):
        model = cls(dictionary['rank'])
        for (a, b, delta), token in dictionary['rules']:
            if token != 256+len(model.rules) or len(delta) != model.rank:
                raise ValueError('invalid rule order or rank')
            atoms = dict(model.shapes[a])
            for coord, value in model.shapes[b].items():
                shifted = add(coord, delta)
                if shifted in atoms:
                    raise ValueError('dictionary has overlapping shapes')
                atoms[shifted] = value
            model.shapes[token] = atoms
            model.rules.append(((a, b, tuple(delta)), token))
        return model

    def without_anchors(self, packet, cells):
        points = list(product(*(range(n) for n in packet['shape'])))
        mask = None
        if len(cells) != len(points):
            bits = bytearray((len(points)+7)//8)
            for index, point in enumerate(points):
                if point in cells:
                    bits[index//8] |= 1 << (index%8)
            mask = base64.b64encode(bits).decode('ascii')
        return {'shape':packet['shape'], 'tokens':[token for token, _ in packet['tokens']], 'mask':mask}

    def decode_without_anchors(self, packet):
        shape = tuple(packet['shape'])
        points = list(product(*(range(n) for n in shape)))
        if packet['mask'] is not None:
            bits = base64.b64decode(packet['mask'], validate=True)
            if len(bits) != (len(points)+7)//8:
                raise ValueError('invalid mask length')
            points = [point for i,point in enumerate(points) if bits[i//8] & (1 << (i%8))]
        domain = set(points)
        cells = {}
        index = 0
        for token in packet['tokens']:
            while index < len(points) and points[index] in cells:
                index += 1
            if index == len(points):
                raise ValueError('extra token after domain covered')
            anchor = points[index]
            for offset, value in self.shapes[token].items():
                coord = add(anchor, offset)
                if coord in cells or coord not in domain:
                    raise ValueError('overlapping or out-of-domain cell')
                cells[coord] = value
        if set(cells) != domain:
            raise ValueError('incomplete domain coverage')
        return shape, cells


def sample(shape, phase=0, holes=False):
    cells = {}
    for point in product(*(range(n) for n in shape)):
        if holes and sum(point) % 5 == 1:
            continue
        cells[point] = (sum(point)+phase) % 4
    return tuple(shape), cells


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if os.name=='nt':
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
        assert psutil.Process().nice()==psutil.IDLE_PRIORITY_CLASS
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    results=[]
    for shape, holes in [((16,),False),((3,7),False),((4,4,4),True),((2,3,2,3),False),((2,)*8,True)]:
        train=[sample(shape,phase,holes) for phase in range(3)]
        model=AxisBPE(len(shape));model.learn(train)
        frozen=json.dumps(model.dictionary(),sort_keys=True)
        test_shape, cells=sample(shape,3,holes)
        packet=model.encode(test_shape,cells)
        loaded=AxisBPE.from_dictionary(json.loads(frozen))
        rebuilt=loaded.decode(json.loads(json.dumps(packet)))
        assert rebuilt==(test_shape,cells)
        compact=model.without_anchors(packet,cells)
        assert loaded.decode_without_anchors(json.loads(json.dumps(compact)))==(test_shape,cells)
        assert json.dumps(model.dictionary(),sort_keys=True)==frozen
        results.append({'shape':shape,'holes':holes,'base_cells':len(cells),'rules':len(model.rules),
            'tokens':len(packet['tokens']),'ratio':len(packet['tokens'])/len(cells),
            'anchor_integer_count':len(shape)*len(packet['tokens']),
            'dictionary_json_bytes':len(frozen.encode()),'packet_json_bytes':len(json.dumps(packet).encode()),
            'raw_cells_json_bytes':len(json.dumps([[list(k),v] for k,v in sorted(cells.items())]).encode()),
            'compact_packet_json_bytes':len(json.dumps(compact).encode()),
            'raw_uint8_bytes':len(cells),'shape_cells':math.prod(shape),
            'mask_raw_bytes':(math.prod(shape)+7)//8 if holes else 0,
            'lossless':True,'inference_dictionary_unchanged':True})
    a=((2,2),{(0,0):1,(0,1):2,(1,0):3,(1,1):4})
    b=((2,2),{(0,0):1,(0,1):3,(1,0):2,(1,1):4})
    model=AxisBPE(2);model.learn([a,b,a,b],budget=1)
    pa=model.encode(*a);pb=model.encode(*b)
    assert pa!=pb and model.decode(pa)==a and model.decode(pb)==b
    # Property-style coverage: random shapes, sparse masks, and unseen pixel values.
    rng=random.Random(20261111)
    for rank in range(1,6):
        for _ in range(8):
            shape=tuple(rng.randint(1,4) for _ in range(rank))
            model=AxisBPE(rank);model.learn([sample(shape,p,True) for p in range(3)],budget=6)
            cells={p:rng.randrange(256) for p in product(*(range(n) for n in shape)) if rng.random()>.25}
            encoded=model.encode(shape,cells)
            assert model.decode(encoded)==(shape,cells)
            compact=model.without_anchors(encoded,cells)
            assert model.decode_without_anchors(compact)==(shape,cells)
    report={'results':results,'horizontal_vertical_distinct':True,'collision_packets':[pa,pb],
            'random_sparse_cases':40,'seconds':time.perf_counter()-started,
            'scope':'representation prototype; metadata explicit, no compression superiority claim'}
    (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
