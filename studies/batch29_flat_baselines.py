"""Batch 29: plain 1D orders against the 2D candidate, plus a 3D check.

Does not change existing script defaults or official code.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMBA_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import json
import math
import pickle
import tarfile
import time
import zlib
from collections import Counter
from pathlib import Path

import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
import numpy as np
import torch
torch.set_num_threads(1)

from lossless_nd_candidate import LosslessNDCandidate
from nd_bpe_incremental_probe import Index


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def h0(values):
    counts = Counter(values)
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def id_bits(size):
    return max(1, math.ceil(math.log2(max(size, 2))))


def conditional_bits(sequences):
    pairs = Counter()
    prevs = Counter()
    firsts = []
    tokens = 0
    for seq in sequences:
        if not seq:
            continue
        firsts.append(seq[0])
        tokens += len(seq)
        for left, right in zip(seq, seq[1:]):
            pairs[(left, right)] += 1
            prevs[left] += 1
    cond = 0.0
    for (left, _), count in pairs.items():
        probability = count / prevs[left]
        cond += count * (-math.log2(probability))
    return h0(firsts) * len(firsts) + cond, tokens


def hilbert_order(n=32):
    def point(index):
        x = y = 0
        step = 1
        while step < n:
            rx = 1 & (index // 2)
            ry = 1 & (index ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = step - 1 - x, step - 1 - y
                x, y = y, x
            x += step * rx
            y += step * ry
            index //= 4
            step *= 2
        return y, x
    order = [point(index) for index in range(n * n)]
    assert len(set(order)) == n * n
    return order


ORDERS = {
    'row': [(y, x) for y in range(32) for x in range(32)],
    'column': [(y, x) for x in range(32) for y in range(32)],
    'snake': [(y, x if y % 2 == 0 else 31 - x) for y in range(32) for x in range(32)],
    'hilbert': hilbert_order(),
}


def flatten(image, order):
    return [int(image[y, x]) for y, x in order]


def cells_2d(image):
    return {(int(y), int(x)): int(image[y, x]) for y in range(32) for x in range(32)}


def volume(rgb):
    cells = {(y, x, channel): int(rgb[channel, y, x])
             for channel in range(3) for y in range(32) for x in range(32)}
    return (32, 32, 3), cells


def interleave(rgb):
    return [int(rgb[channel, y, x]) for y in range(32) for x in range(32) for channel in range(3)]


def concat_channels(rgb):
    return [int(rgb[channel, y, x]) for channel in range(3) for y in range(32) for x in range(32)]


def learn_plain(sequences, budget):
    current = [list(seq) for seq in sequences]
    rules = []
    next_id = 256
    for _ in range(budget):
        counts = Counter()
        for seq in current:
            counts.update(zip(seq, seq[1:]))
        if not counts:
            break
        pair = min(counts, key=lambda item: (-counts[item], item))
        if counts[pair] < 2:
            break
        rules.append((pair, next_id))
        merged = []
        left, right = pair
        for seq in current:
            out = []
            index = 0
            size = len(seq)
            while index < size:
                if index + 1 < size and seq[index] == left and seq[index + 1] == right:
                    out.append(next_id)
                    index += 2
                else:
                    out.append(seq[index])
                    index += 1
            merged.append(out)
        current = merged
        next_id += 1
    return rules


def apply_prefixes(sequences, rules, checkpoints):
    current = [list(seq) for seq in sequences]
    originals = [list(seq) for seq in sequences]
    shapes = {value: [value] for value in range(256)}
    saved = {}
    for step, ((left, right), token) in enumerate(rules, start=1):
        shapes[token] = shapes[left] + shapes[right]
        nxt = []
        for seq in current:
            out = []
            index = 0
            size = len(seq)
            while index < size:
                if index + 1 < size and seq[index] == left and seq[index + 1] == right:
                    out.append(token)
                    index += 2
                else:
                    out.append(seq[index])
                    index += 1
            nxt.append(out)
        current = nxt
        if step in checkpoints or step == len(rules):
            bad = []
            for index, seq in enumerate(current):
                flat = []
                for item in seq:
                    flat.extend(shapes[item])
                if flat != originals[index]:
                    bad.append(index)
            saved[step] = (current, bad)
    return saved


def plain_cost(encoded, rules, pixels, raw):
    tokens = [token for seq in encoded for token in seq]
    width = id_bits(256 + len(rules))
    dictionary = 256 + len(rules) * (2 * width)
    images = len(encoded)
    side = images * 16 + images
    total = h0(tokens) * len(tokens) + dictionary + side
    cond, _ = conditional_bits(encoded)
    cond_total = cond + dictionary + side
    return {'tokens': len(tokens), 'ratio': len(tokens) / pixels, 'actual_rules': len(rules),
            'bits_per_pixel': total / pixels, 'token_bits_per_pixel': (h0(tokens) * len(tokens)) / pixels,
            'conditional_bits_per_pixel': cond_total / pixels, 'dictionary_bits': dictionary,
            'raw_zero_order_bits_per_pixel': (h0(raw) * len(raw)) / pixels,
            'zlib9_bits_per_pixel': (len(zlib.compress(raw, 9)) * 8) / pixels}


def direction(delta):
    dx, dy = delta
    if dx == 0 and dy > 0:
        return 'horizontal'
    if dy == 0 and dx > 0:
        return 'vertical'
    return 'other'


def learn_candidate(rank, samples, budget):
    model = LosslessNDCandidate(rank)
    model.learn(samples, budget=budget, score='frequency')
    return model


def slice_model(model, budget):
    data = model.dictionary()
    data = dict(data)
    data['rules'] = data['rules'][:budget]
    return LosslessNDCandidate.from_dictionary(data)


def evaluate_candidate(model, samples):
    clone = LosslessNDCandidate.from_dictionary(json.loads(json.dumps(model.dictionary())))
    encoded = []
    bad = []
    literals = 0
    for index, (shape, cells) in enumerate(samples):
        packet = model.encode(shape, cells)
        if clone.decode(json.loads(json.dumps(packet))) != (tuple(shape), cells):
            bad.append(index)
        encoded.append(packet['tokens'])
        literals += len(packet['literals'])
    return encoded, bad, literals


def candidate_cost(encoded, model, literals, pixels, raw):
    tokens = [token for seq in encoded for token in seq]
    dictionary = model.dictionary()
    vocab = len(dictionary['base']) + len(dictionary['rules'])
    width = id_bits(vocab)
    rank = dictionary['rank']
    dictionary_bits = 256 + len(dictionary['rules']) * (2 * width + rank * 6)
    images = len(encoded)
    side = images * (8 * rank) + images + literals * (width + 8)
    zero = h0(tokens) * len(tokens)
    cond, _ = conditional_bits(encoded)
    return {'tokens': len(tokens), 'ratio': len(tokens) / pixels, 'actual_rules': len(dictionary['rules']),
            'bits_per_pixel': (zero + dictionary_bits + side) / pixels,
            'token_bits_per_pixel': zero / pixels,
            'conditional_bits_per_pixel': (cond + dictionary_bits + side) / pixels,
            'dictionary_bits': dictionary_bits,
            'raw_zero_order_bits_per_pixel': (h0(raw) * len(raw)) / pixels,
            'zlib9_bits_per_pixel': (len(zlib.compress(raw, 9)) * 8) / pixels,
            'literals': literals}


def usage(model, samples):
    inventory = Counter()
    for (_left, _right, delta), _token in model.rules:
        inventory[direction(delta)] += 1
        inventory['unit_' + direction(delta)] += int(abs(delta[0]) + abs(delta[1]) == 1)
    fired = Counter()
    for shape, cells in samples:
        base = {value: index for index, value in enumerate(model.base)}
        unknown = sorted(set(cells.values()) - set(base))
        literals = {len(model.shapes) + index: value for index, value in enumerate(unknown)}
        ids = {**base, **{value: index for index, value in literals.items()}}
        index = Index()
        state = model._state(shape, {point: ids[value] for point, value in cells.items()},
                             index, 0, len(model.shapes) + len(unknown))
        for key, token in model.rules:
            kind = direction(key[2])
            found = []
            for _, left, right in index.occurrences.get(key, ()):
                if state.regions[left][1] > state.regions[right][1]:
                    left, right = right, left
                found.append((state.edges[left][right], left, right))
            for _, left, right in sorted(found):
                fired[kind] += state.replace(left, right, token)
    return {'rules': dict(inventory), 'test_replacements': dict(fired)}


def load_dats(root):
    files = sorted(root.rglob('*.dat'), key=lambda path: int(path.stem))
    return [torch.load(path, map_location='cpu', weights_only=False).long().numpy() for path in files]


def class_indices(labels, per_class):
    chosen = []
    for label in range(10):
        found = np.flatnonzero(labels == label)[:per_class]
        chosen.extend(int(index) for index in found)
    return chosen


def run_image_suite(name, train_images, test_sets, out):
    """train_images: list of 32x32 arrays. test_sets: name -> (arrays, raw bytes)."""
    report = {'dataset': name, 'train_images': len(train_images), 'orders': {}, 'candidate': {}}
    started = time.perf_counter()
    checkpoints = (256, 512, 1024)
    flat_train = {order_name: [flatten(image, order) for image in train_images] for order_name, order in ORDERS.items()}
    learned = {order_name: learn_plain(rows, 1024) for order_name, rows in flat_train.items()}
    flat_tests = {order_name: {test_name: [flatten(image, ORDERS[order_name]) for image in images]
                               for test_name, (images, _raw) in test_sets.items()}
                  for order_name in ORDERS}
    applied = {order_name: {test_name: apply_prefixes(rows, learned[order_name], checkpoints)
                            for test_name, rows in tests.items()}
               for order_name, tests in flat_tests.items()}
    for order_name in ORDERS:
        order_rows = {}
        for budget in checkpoints:
            rules = learned[order_name][:budget]
            budget_rows = {}
            for test_name, (images, raw) in test_sets.items():
                encoded, bad = applied[order_name][test_name].get(budget) or applied[order_name][test_name][max(applied[order_name][test_name])]
                cost = plain_cost(encoded, rules, len(images) * 1024, raw)
                cost['lossless_bad'] = len(bad)
                budget_rows[test_name] = cost
            order_rows[str(budget)] = budget_rows
            print(name, order_name, budget, {key: round(row['ratio'], 6) for key, row in budget_rows.items()}, flush=True)
        report['orders'][order_name] = order_rows
        save(out, report)
    model = learn_candidate(2, [((32, 32), cells_2d(image)) for image in train_images], 1024)
    for budget in (256, 512, 1024):
        part = slice_model(model, budget)
        budget_rows = {}
        for test_name, (images, raw) in test_sets.items():
            samples = [((32, 32), cells_2d(image)) for image in images]
            encoded, bad, literals = evaluate_candidate(part, samples)
            cost = candidate_cost(encoded, part, literals, len(images) * 1024, raw)
            cost['lossless_bad'] = len(bad)
            if test_name == list(test_sets)[-1]:
                cost['directions'] = usage(part, samples)
            budget_rows[test_name] = cost
        report['candidate'][str(budget)] = budget_rows
        save(out, report)
        print(name, '2d', budget, {key: round(row['ratio'], 6) for key, row in budget_rows.items()}, flush=True)
    report['seconds'] = time.perf_counter() - started
    save(out, report)
    return report


def run_color(pixels, indices, paired, out):
    train = [pixels[index] for index in indices]
    test = []
    for position in range(100):
        test.append(np.stack([paired[name]['paired100'][position] for name in ('red', 'green', 'blue')]))
    report = {'train_images': 300, 'test_images': 100, 'plain': {}, 'volume_512': None,
              'volume_256_from_batch28': {'ratio': 286510 / (100 * 3072), 'tokens': 286510,
                                          'lossless': '100/100', 'budget': 256}}
    for order_name, function in (('interleaved', interleave), ('channel_concat', concat_channels)):
        learned = learn_plain([function(image) for image in train], 512)
        report['plain'][order_name] = {}
        for budget in (256, 512):
            saved = apply_prefixes([function(image) for image in test], learned[:budget], (budget,))
            encoded, bad = saved.get(budget) or saved[max(saved)]
            raw = np.ascontiguousarray(np.stack(test).astype(np.uint8)).tobytes()
            cost = plain_cost(encoded, learned[:budget], 100 * 3072, raw)
            cost['lossless_bad'] = len(bad)
            report['plain'][order_name][str(budget)] = cost
            print('P13', order_name, budget, cost['ratio'], cost['lossless_bad'], flush=True)
        save(out, report)
    volume_model = learn_candidate(3, [volume(image) for image in train], 512)
    raw = np.ascontiguousarray(np.stack(test).astype(np.uint8)).tobytes()
    encoded, bad, literals = evaluate_candidate(volume_model, [volume(image) for image in test])
    cost = candidate_cost(encoded, volume_model, literals, 100 * 3072, raw)
    cost['lossless_bad'] = len(bad)
    report['volume_512'] = cost
    save(out, report)
    print('P13 volume512', cost['ratio'], cost['lossless_bad'], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--only', choices=('all', 'p13'), default='all')
    parser.add_argument('--svhn-test-grids', type=Path)
    parser.add_argument('--celeba-test-grids', type=Path)
    args = parser.parse_args()
    if args.only == 'all' and (args.svhn_test_grids is None or args.celeba_test_grids is None):
        parser.error('--only all requires --svhn-test-grids and --celeba-test-grids')
    repo = Path(__file__).resolve().parents[1]
    meta = json.loads((args.inputs / 'input.json').read_text(encoding='utf-8'))
    channels = {}
    for name, row in meta['channels'].items():
        channels[name] = {'test1000': np.load(row['files']['test1000']['path']),
                          'paired100': np.load(row['files']['paired100']['path'])}
    with tarfile.open(repo / 'data/cifar-10-python.tar.gz') as archive:
        batch = pickle.load(archive.extractfile('cifar-10-batches-py/data_batch_1'), encoding='bytes')
    labels = np.asarray(batch[b'labels'])
    pixels = np.asarray(batch[b'data']).reshape(-1, 3, 32, 32)
    indices = class_indices(labels, 30)
    assert sorted(class_indices(labels, 1)) == sorted(meta['train_indices'])
    if args.only == 'p13':
        run_color(pixels, indices, channels, args.out / 'p13-color' / 'report.json')
        print('BATCH29_DONE', flush=True)
        return
    for name, axis in (('red', 0), ('green', 1), ('blue', 2)):
        train = list(pixels[indices, axis])
        tests = {
            'paired100': (channels[name]['paired100'], channels[name]['paired100'].astype(np.uint8).tobytes()),
            'test1000': (channels[name]['test1000'], channels[name]['test1000'].astype(np.uint8).tobytes()),
        }
        run_image_suite(name, train, tests, args.out / 'p12-cifar' / f'{name}.json')
    suites = {
        'svhn': (args.svhn_test_grids,
                 repo / 'results/2026-11-10-elsner-color-modes/svhn/gray/train/transcribed_data_og'),
        'celeba': (args.celeba_test_grids,
                   repo / 'results/2026-11-10-elsner-color-modes/celeba/gray/train/transcribed_data_og'),
    }
    for name, (test_root, train_root) in suites.items():
        train = load_dats(train_root)
        test = load_dats(test_root)
        raw = np.stack(test).astype(np.uint8).tobytes()
        run_image_suite(name, train, {name: (test, raw)}, args.out / 'p12-other' / f'{name}.json')
    run_color(pixels, indices, channels, args.out / 'p13-color' / 'report.json')
    print('BATCH29_DONE', flush=True)


if __name__ == '__main__':
    main()
