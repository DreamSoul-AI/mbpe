"""Exhaustive small-domain and seeded variable-shape candidate checks."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import time

from lossless_nd_candidate import LosslessNDCandidate


def check(model, shape, cells):
    dictionary = json.dumps(model.dictionary(), sort_keys=True)
    packet = json.loads(json.dumps(model.encode(shape, cells)))
    clone = LosslessNDCandidate.from_dictionary(json.loads(dictionary))
    assert clone.decode(packet) == (shape, cells)
    assert json.dumps(model.dictionary(), sort_keys=True) == dictionary
    return json.dumps(packet, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import psutil
    proc = psutil.Process()
    proc.nice(psutil.IDLE_PRIORITY_CLASS)
    proc.cpu_affinity([proc.cpu_affinity()[0]])
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = []
    for shape in ((2, 3), (2, 2, 2)):
        points = list(product(*(range(n) for n in shape)))
        training = [(shape, {p: (sum(p)+phase) % 2 for p in points}) for phase in range(4)]
        for score in ('frequency', 'realizable'):
            model = LosslessNDCandidate(len(shape))
            model.learn(training, budget=8, score=score)
            for alphabet in ((None, 0, 1), (None, -1, 2**80)):
                packets = set()
                merged_packets = 0
                for values in product(alphabet, repeat=len(points)):
                    cells = {p: v for p, v in zip(points, values) if v is not None}
                    packet = check(model, shape, cells)
                    assert packet not in packets, 'Distinct domain/value inputs collided'
                    packets.add(packet)
                    if len(json.loads(packet)['tokens']) < len(cells):
                        merged_packets += 1
                rows.append(dict(shape=shape, score=score, cases=len(packets),
                                 losses=0, packet_collisions=0,
                                 atoms=alphabet[1:], merged_packets=merged_packets))
                print('EXHAUSTIVE', shape, score, len(packets), 'merged', merged_packets, flush=True)
    rng = random.Random(20261111)
    for rank in (0, 1, 2, 3, 4, 8):
        train_shape = (2,) * rank
        points = list(product(*(range(n) for n in train_shape)))
        training = [(train_shape, {p: (sum(p)+phase) % 3-1 for p in points}) for phase in range(3)]
        model = LosslessNDCandidate(rank)
        model.learn(training, budget=12)
        for _ in range(100):
            shape = tuple(rng.randrange(4) for _ in range(rank))
            cells = {p: rng.choice((-2**90, -1, 0, 1, 2**80))
                     for p in product(*(range(n) for n in shape)) if rng.random() < .6}
            check(model, shape, cells)
        rows.append(dict(rank=rank, random_cases=100, losses=0))
    report = dict(seed=20261111, rows=rows, seconds=time.perf_counter()-started,
                  candidate_sha256=hashlib.sha256(Path(__file__).with_name('lossless_nd_candidate.py').read_bytes()).hexdigest())
    (args.out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
