"""Batch 31: MNIST masks, four 1D orders against the 2D candidate.

Does not change existing script defaults or official code.
28 and 56 are not powers of two, so Hilbert order is the next power-of-two
Hilbert curve restricted to the cells that lie inside the image.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMBA_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import json
import struct
import time
from collections import Counter
from pathlib import Path

import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
import numpy as np

from batch30_mask_baselines import (
    CHECKPOINTS, blobs, candidate_cost, cells_2d, direction, evaluate_candidate,
    flatten, learn_apply, mask_image, pick, plain_cost, save, slice_model,
)
from lossless_nd_candidate import LosslessNDCandidate
from nd_bpe_incremental_probe import Index

RAW = Path(__file__).resolve().parents[1] / 'third_party' / 'MDBPE_TF' / 'data' / 'MNIST' / 'raw'
SEED = 20261112


def hilbert_order(n):
    side = 1
    while side < n:
        side *= 2

    def point(index):
        x = y = 0
        step = 1
        while step < side:
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

    order = []
    for index in range(side * side):
        y, x = point(index)
        if y < n and x < n:
            order.append((y, x))
    assert len(order) == n * n and len(set(order)) == n * n
    return order


def make_orders(n):
    return {
        'row': [(y, x) for y in range(n) for x in range(n)],
        'column': [(y, x) for x in range(n) for y in range(n)],
        'snake': [(y, x if y % 2 == 0 else n - 1 - x) for y in range(n) for x in range(n)],
        'hilbert': hilbert_order(n),
    }


def load_idx(path):
    blob = Path(path).read_bytes()
    magic, count, rows, cols = struct.unpack('>IIII', blob[:16])
    if magic != 2051:
        raise SystemExit(f'unexpected idx magic {magic}')
    data = np.frombuffer(blob, dtype=np.uint8, offset=16).reshape(count, rows, cols)
    return np.array(data, copy=True)


def choose_indices(count, size):
    rng = np.random.default_rng(SEED)
    return [int(index) for index in rng.choice(count, size=size, replace=False)]


def classify(shape):
    points = list(shape)
    ys = [point[0] for point in points]
    xs = [point[1] for point in points]
    height = max(ys) - min(ys) + 1
    width = max(xs) - min(xs) + 1
    area = len(points)
    if min(height, width) == 1:
        kind = 'strip'
    elif area * 2 >= height * width:
        kind = 'compact'
    else:
        kind = 'other'
    return kind, area, height, width


def shape_usage(model, samples, checkpoints):
    classes = []
    inventory = Counter()
    inventories = {}
    summaries = {}
    for step, ((_left, _right, delta), token) in enumerate(model.rules, start=1):
        inventory[direction(delta)] += 1
        inventory['unit_' + direction(delta)] += int(abs(delta[0]) + abs(delta[1]) == 1)
        kind, area, height, width = classify(model.shapes[token])
        classes.append((kind, area, height, width))
        if step in checkpoints or step == len(model.rules):
            inventories[step] = dict(inventory)
            picked = classes[:step]
            areas = [item[1] for item in picked]
            summaries[step] = {
                'strip': sum(item[0] == 'strip' for item in picked),
                'compact': sum(item[0] == 'compact' for item in picked),
                'other': sum(item[0] == 'other' for item in picked),
                'area_ge_4': sum(area >= 4 for area in areas),
                'area_ge_8': sum(area >= 8 for area in areas),
                'area_ge_16': sum(area >= 16 for area in areas),
                'median_area': float(np.median(areas)) if areas else 0,
                'max_area': max(areas) if areas else 0,
                'max_side': max(max(item[2], item[3]) for item in picked) if picked else 0,
            }
    fired = Counter()
    shape_fired = Counter()
    fired_at = {}
    shape_at = {}
    prepared = []
    for shape, cells in samples:
        base = {value: index for index, value in enumerate(model.base)}
        unknown = sorted(set(cells.values()) - set(base))
        literals = {len(model.shapes) + index: value for index, value in enumerate(unknown)}
        ids = {**base, **{value: index for index, value in literals.items()}}
        index = Index()
        state = model._state(shape, {point: ids[value] for point, value in cells.items()},
                             index, 0, len(model.shapes) + len(unknown))
        prepared.append((state, index))
    for step, (key, token) in enumerate(model.rules, start=1):
        kind = direction(key[2])
        shape_kind = classes[step - 1][0]
        for state, index in prepared:
            found = []
            for _, left, right in index.occurrences.get(key, ()):
                if state.regions[left][1] > state.regions[right][1]:
                    left, right = right, left
                found.append((state.edges[left][right], left, right))
            for _, left, right in sorted(found):
                replaced = state.replace(left, right, token)
                fired[kind] += replaced
                shape_fired[shape_kind] += replaced
        if step in checkpoints or step == len(model.rules):
            fired_at[step] = dict(fired)
            shape_at[step] = dict(shape_fired)
            summaries[step]['replacements'] = dict(shape_fired)
    return inventories, fired_at, summaries


def run_suite(name, train_images, test_sets, kind, orders_wanted, out):
    from batch30_mask_baselines import alphabet_of
    alphabet = alphabet_of(kind)
    side = train_images[0].shape[0]
    orders = make_orders(side)
    pixels_of = {test_name: len(images) * side * side
                 for test_name, (images, _flat, _byte, _packed) in test_sets.items()}
    report = {'dataset': name, 'mask': kind, 'alphabet': alphabet, 'train_images': len(train_images),
              'side': side, 'hilbert': 'next power of two, restricted to the image', 'orders': {}, 'candidate': {}}
    started = time.perf_counter()
    for order_name in orders_wanted:
        if order_name == '2d':
            continue
        order = orders[order_name]
        train = [flatten(image, order) for image in train_images]
        tests = {test_name: [flatten(image, order) for image in images]
                 for test_name, (images, _flat, _byte, _packed) in test_sets.items()}
        rules, saved = learn_apply(train, tests, alphabet, 1024, set(CHECKPOINTS))
        order_rows = {}
        for budget in CHECKPOINTS:
            budget_rows = {}
            for test_name, (_images, flat, byte_blob, packed_blob) in test_sets.items():
                (encoded, bad), step = pick(saved[test_name], budget)
                rule_count = min(budget, len(rules))
                if step != rule_count and step != budget:
                    rule_count = step
                cost = plain_cost(encoded, rule_count, alphabet, pixels_of[test_name], flat, byte_blob, packed_blob)
                cost['lossless_bad'] = len(bad)
                budget_rows[test_name] = cost
            order_rows[str(budget)] = budget_rows
            print(name, kind, order_name, budget,
                  {key: round(row['ratio'], 6) for key, row in budget_rows.items()},
                  {key: row['actual_rules'] for key, row in budget_rows.items()},
                  {key: row['lossless_bad'] for key, row in budget_rows.items()}, flush=True)
        report['orders'][order_name] = order_rows
        save(out, report)
    if '2d' in orders_wanted or orders_wanted == ('row', 'column', 'snake', 'hilbert', '2d'):
        pass
    samples_train = [((side, side), cells_2d(image)) for image in train_images]
    model = LosslessNDCandidate(2)
    model.learn(samples_train, budget=1024, score='frequency')
    last_name = list(test_sets)[-1]
    direction_samples = [((side, side), cells_2d(image)) for image in test_sets[last_name][0]]
    inventories, fired_at, summaries = shape_usage(model, direction_samples, set(CHECKPOINTS))
    for budget in CHECKPOINTS:
        part = slice_model(model, budget)
        budget_rows = {}
        for test_name, (images, flat, byte_blob, packed_blob) in test_sets.items():
            samples = [((side, side), cells_2d(image)) for image in images]
            encoded, bad, literals = evaluate_candidate(part, samples)
            cost = candidate_cost(encoded, part, literals, alphabet, pixels_of[test_name], flat, byte_blob, packed_blob)
            cost['lossless_bad'] = len(bad)
            if test_name == last_name:
                step = cost['actual_rules']
                cost['directions'] = {
                    'rules': inventories.get(step, {}),
                    'test_replacements': fired_at.get(step, {}),
                }
                cost['shapes'] = summaries.get(step, {})
            budget_rows[test_name] = cost
        report['candidate'][str(budget)] = budget_rows
        save(out, report)
        print(name, kind, '2d', budget,
              {key: round(row['ratio'], 6) for key, row in budget_rows.items()},
              {key: row['actual_rules'] for key, row in budget_rows.items()},
              {key: row['lossless_bad'] for key, row in budget_rows.items()}, flush=True)
    report['seconds'] = time.perf_counter() - started
    save(out, report)


def prepare(images, kind):
    masked = [mask_image(image, kind) for image in images]
    from batch30_mask_baselines import alphabet_of
    flat, byte_blob, packed = blobs(masked, alphabet_of(kind))
    return masked, flat, byte_blob, packed


def upsample(images):
    return [np.repeat(np.repeat(image, 2, axis=0), 2, axis=1) for image in images]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mask', choices=('bin128', '4level'), required=True)
    parser.add_argument('--train-sizes', default='300,1000')
    parser.add_argument('--orders', default='row,column,snake,hilbert')
    parser.add_argument('--upsample', action='store_true')
    parser.add_argument('--tag', default='')
    parser.add_argument('--test-count', type=int, default=1000)
    args = parser.parse_args()
    train_all = load_idx(RAW / 'train-images-idx3-ubyte')
    test_all = load_idx(RAW / 't10k-images-idx3-ubyte')
    chosen = choose_indices(len(train_all), 1000)
    test_images = list(test_all[:args.test_count])
    orders = tuple(item.strip() for item in args.orders.split(',') if item.strip())
    for size_text in args.train_sizes.split(','):
        size = int(size_text)
        train_raw = [train_all[index] for index in chosen[:size]]
        test_raw = test_images
        if args.upsample:
            train_raw = upsample(train_raw)
            test_raw = upsample(test_raw)
        train, flat, byte_blob, packed = prepare(train_raw, args.mask)
        _test, tflat, tbyte, tpacked = prepare(test_raw, args.mask)
        folder = 'p24-up' if args.upsample else 'p21'
        suffix = '' if args.test_count == 1000 else f'-test{args.test_count}'
        name = f'{args.mask}-train{size}{suffix}{args.tag}'
        run_suite(name, train, {'test1000': (_test, tflat, tbyte, tpacked)}, args.mask, orders + ('2d',),
                  args.out / folder / args.mask / f'train{size}{suffix}.json')
        meta = {'seed': SEED, 'train_indices': chosen[:size],
                'test': f'first {args.test_count} of t10k', 'upsample': args.upsample}
        save(args.out / folder / args.mask / f'train{size}{suffix}-indices.json', meta)
    print('BATCH31_PART_DONE', args.mask, 'up' if args.upsample else 'native', flush=True)


if __name__ == '__main__':
    main()
