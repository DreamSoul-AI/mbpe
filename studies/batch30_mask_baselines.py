"""Batch 30: plain 1D orders against the 2D candidate on small-alphabet masks.

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

from lossless_nd_candidate import LosslessNDCandidate
from nd_bpe_incremental_probe import Index


CHECKPOINTS = (64, 256, 1024)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def h0_values(values):
    if len(values) == 0:
        return 0.0
    counts = np.bincount(np.asarray(values, dtype=np.int64))
    total = int(counts.sum())
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def id_bits(size):
    return max(1, math.ceil(math.log2(max(size, 2))))


def conditional_bits(sequences):
    pairs = Counter()
    prevs = Counter()
    firsts = []
    for seq in sequences:
        if not seq:
            continue
        firsts.append(seq[0])
        for left, right in zip(seq, seq[1:]):
            pairs[(left, right)] += 1
            prevs[left] += 1
    cond = 0.0
    for (left, _), count in pairs.items():
        cond += count * (-math.log2(count / prevs[left]))
    return h0_values(firsts) * len(firsts) + cond


def hilbert_order(n):
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


def make_orders(n):
    return {
        'row': [(y, x) for y in range(n) for x in range(n)],
        'column': [(y, x) for x in range(n) for y in range(n)],
        'snake': [(y, x if y % 2 == 0 else n - 1 - x) for y in range(n) for x in range(n)],
        'hilbert': hilbert_order(n),
    }


def flatten(image, order):
    return np.asarray([int(image[y, x]) for y, x in order], dtype=np.int32)


def cells_2d(image):
    height, width = image.shape
    return {(int(y), int(x)): int(image[y, x]) for y in range(height) for x in range(width)}


def mask_image(image, kind):
    image = np.asarray(image)
    if kind == 'bin128':
        return (image >= 128).astype(np.int64)
    if kind == 'bin_permedian':
        return (image >= np.median(image)).astype(np.int64)
    if kind == '4level':
        return (image // 64).astype(np.int64)
    raise ValueError(kind)


def alphabet_of(kind):
    return 4 if kind == '4level' else 2


def pack_cells(flat, alphabet):
    flat = np.asarray(flat, dtype=np.uint8)
    if alphabet == 2:
        return np.packbits(flat).tobytes()
    if flat.size % 4:
        flat = np.pad(flat, (0, 4 - flat.size % 4))
    grouped = flat.reshape(-1, 4)
    packed = (grouped[:, 0] << 6) | (grouped[:, 1] << 4) | (grouped[:, 2] << 2) | grouped[:, 3]
    return packed.astype(np.uint8).tobytes()


def merge_one(seq, left, right, token):
    if seq.size < 2:
        return seq
    hits = np.flatnonzero((seq[:-1] == left) & (seq[1:] == right))
    if hits.size == 0:
        return seq
    take = []
    last = -2
    for pos in hits.tolist():
        if pos > last + 1:
            take.append(pos)
            last = pos
    take = np.asarray(take, dtype=np.int64)
    out = seq.copy()
    out[take] = token
    keep = np.ones(out.size, dtype=bool)
    keep[take + 1] = False
    return out[keep]


def learn_apply(train, tests, alphabet, budget, checkpoints):
    """tests: name -> list of int32 arrays. Returns rules and name -> step -> (seqs, bad)."""
    current = [seq.copy() for seq in train]
    applied = {name: [seq.copy() for seq in rows] for name, rows in tests.items()}
    originals = {name: [seq.copy() for seq in rows] for name, rows in tests.items()}
    shapes = {value: np.asarray([value], dtype=np.int32) for value in range(alphabet)}
    rules = []
    saved = {name: {} for name in tests}

    def snapshot(step):
        for name, rows in applied.items():
            bad = []
            encoded = []
            for index, seq in enumerate(rows):
                if seq.size:
                    flat = np.concatenate([shapes[int(token)] for token in seq.tolist()])
                else:
                    flat = np.asarray([], dtype=np.int32)
                if flat.size != originals[name][index].size or not np.array_equal(flat, originals[name][index]):
                    bad.append(index)
                encoded.append(seq.astype(int).tolist())
            saved[name][step] = (encoded, bad)

    for step in range(1, budget + 1):
        width = alphabet + step - 1
        counts = np.zeros((width, width), dtype=np.int32)
        for seq in current:
            if seq.size < 2:
                continue
            np.add.at(counts, (seq[:-1], seq[1:]), 1)
        flat = int(np.argmax(counts))
        if int(counts.flat[flat]) < 2:
            break
        left, right = divmod(flat, width)
        token = alphabet + step - 1
        rules.append((int(left), int(right), int(token)))
        shapes[token] = np.concatenate([shapes[left], shapes[right]])
        current = [merge_one(seq, left, right, token) for seq in current]
        for name in applied:
            applied[name] = [merge_one(seq, left, right, token) for seq in applied[name]]
        if step in checkpoints:
            snapshot(step)
    if not saved[next(iter(saved))]:
        snapshot(len(rules))
    elif len(rules) not in saved[next(iter(saved))]:
        snapshot(len(rules))
    return rules, saved


def pick(saved, budget):
    if budget in saved:
        return saved[budget], budget
    usable = [step for step in saved if step <= budget]
    step = max(usable) if usable else max(saved)
    return saved[step], step


def plain_cost(encoded, rule_count, alphabet, pixels, cells, byte_blob, packed_blob):
    tokens = [token for seq in encoded for token in seq]
    width = id_bits(alphabet + rule_count)
    dictionary = alphabet + rule_count * (2 * width)
    images = len(encoded)
    side = images * 16 + images
    zero = h0_values(tokens) * len(tokens)
    cond = conditional_bits(encoded)
    return {
        'tokens': len(tokens),
        'ratio': len(tokens) / pixels,
        'actual_rules': rule_count,
        'bits_per_cell': (zero + dictionary + side) / pixels,
        'token_bits_per_cell': zero / pixels,
        'conditional_bits_per_cell': (cond + dictionary + side) / pixels,
        'dictionary_bits': dictionary,
        'raw_zero_order_bits_per_cell': h0_values(cells),
        'zlib9_byte_bits_per_cell': (len(zlib.compress(byte_blob, 9)) * 8) / pixels,
        'zlib9_packed_bits_per_cell': (len(zlib.compress(packed_blob, 9)) * 8) / pixels,
    }


def candidate_cost(encoded, model, literals, alphabet, pixels, cells, byte_blob, packed_blob):
    tokens = [token for seq in encoded for token in seq]
    dictionary = model.dictionary()
    vocab = len(dictionary['base']) + len(dictionary['rules'])
    width = id_bits(max(vocab, alphabet))
    rank = dictionary['rank']
    dictionary_bits = alphabet + len(dictionary['rules']) * (2 * width + rank * 6)
    images = len(encoded)
    side = images * (8 * rank) + images + literals * (width + 8)
    zero = h0_values(tokens) * len(tokens)
    cond = conditional_bits(encoded)
    return {
        'tokens': len(tokens),
        'ratio': len(tokens) / pixels,
        'actual_rules': len(dictionary['rules']),
        'bits_per_cell': (zero + dictionary_bits + side) / pixels,
        'token_bits_per_cell': zero / pixels,
        'conditional_bits_per_cell': (cond + dictionary_bits + side) / pixels,
        'dictionary_bits': dictionary_bits,
        'raw_zero_order_bits_per_cell': h0_values(cells),
        'zlib9_byte_bits_per_cell': (len(zlib.compress(byte_blob, 9)) * 8) / pixels,
        'zlib9_packed_bits_per_cell': (len(zlib.compress(packed_blob, 9)) * 8) / pixels,
        'literals': literals,
    }


def direction(delta):
    dx, dy = delta
    if dx == 0 and dy > 0:
        return 'horizontal'
    if dy == 0 and dx > 0:
        return 'vertical'
    return 'other'


def slice_model(model, budget):
    data = dict(model.dictionary())
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


def usage_checkpoints(model, samples, checkpoints):
    inventory = Counter()
    inventories = {}
    for step, ((_left, _right, delta), _token) in enumerate(model.rules, start=1):
        inventory[direction(delta)] += 1
        inventory['unit_' + direction(delta)] += int(abs(delta[0]) + abs(delta[1]) == 1)
        if step in checkpoints or step == len(model.rules):
            inventories[step] = dict(inventory)
    fired = Counter()
    fired_at = {}
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
        for state, index in prepared:
            found = []
            for _, left, right in index.occurrences.get(key, ()):
                if state.regions[left][1] > state.regions[right][1]:
                    left, right = right, left
                found.append((state.edges[left][right], left, right))
            for _, left, right in sorted(found):
                fired[kind] += state.replace(left, right, token)
        if step in checkpoints or step == len(model.rules):
            fired_at[step] = dict(fired)
    return inventories, fired_at


def blobs(images, alphabet):
    flat = np.concatenate([np.asarray(image, dtype=np.uint8).ravel() for image in images])
    return flat, flat.tobytes(), pack_cells(flat, alphabet)


def run_suite(name, train_images, test_sets, kind, out):
    alphabet = alphabet_of(kind)
    side = train_images[0].shape[0]
    orders = make_orders(side)
    pixels_of = {test_name: len(images) * side * side for test_name, (images, _flat, _byte, _packed) in test_sets.items()}
    report = {
        'dataset': name,
        'mask': kind,
        'alphabet': alphabet,
        'train_images': len(train_images),
        'side': side,
        'orders': {},
        'candidate': {},
    }
    started = time.perf_counter()
    learned = {}
    applied = {}
    for order_name, order in orders.items():
        train = [flatten(image, order) for image in train_images]
        tests = {test_name: [flatten(image, order) for image in images]
                 for test_name, (images, _flat, _byte, _packed) in test_sets.items()}
        rules, saved = learn_apply(train, tests, alphabet, 1024, set(CHECKPOINTS))
        learned[order_name] = rules
        applied[order_name] = saved
        order_rows = {}
        for budget in CHECKPOINTS:
            budget_rows = {}
            for test_name, (images, flat, byte_blob, packed_blob) in test_sets.items():
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
    samples_train = [((side, side), cells_2d(image)) for image in train_images]
    model = LosslessNDCandidate(2)
    model.learn(samples_train, budget=1024, score='frequency')
    last_name = list(test_sets)[-1]
    last_images = test_sets[last_name][0]
    direction_samples = [((side, side), cells_2d(image)) for image in last_images]
    inventories, fired_at = usage_checkpoints(model, direction_samples, set(CHECKPOINTS))
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
                    'rules': inventories.get(step, inventories.get(max(inventories), {})),
                    'test_replacements': fired_at.get(step, fired_at.get(max(fired_at), {})),
                }
            budget_rows[test_name] = cost
        report['candidate'][str(budget)] = budget_rows
        save(out, report)
        print(name, kind, '2d', budget,
              {key: round(row['ratio'], 6) for key, row in budget_rows.items()},
              {key: row['actual_rules'] for key, row in budget_rows.items()},
              {key: row['lossless_bad'] for key, row in budget_rows.items()}, flush=True)
    report['seconds'] = time.perf_counter() - started
    save(out, report)


def load_dats(root):
    import torch

    torch.set_num_threads(1)
    files = sorted(root.rglob('*.dat'), key=lambda path: int(path.stem))
    return [torch.load(path, map_location='cpu', weights_only=False).long().numpy() for path in files]


def class_indices(labels, per_class):
    chosen = []
    for label in range(10):
        found = np.flatnonzero(labels == label)[:per_class]
        chosen.extend(int(index) for index in found)
    return chosen


def prepare_sets(train_raw, test_raws, kind):
    train = [mask_image(image, kind) for image in train_raw]
    tests = {}
    for name, images in test_raws.items():
        masked = [mask_image(image, kind) for image in images]
        flat, byte_blob, packed_blob = blobs(masked, alphabet_of(kind))
        tests[name] = (masked, flat, byte_blob, packed_blob)
    return train, tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', required=True,
                        help='comma list: red, green, blue, svhn, celeba')
    parser.add_argument('--svhn-test-grids', type=Path)
    parser.add_argument('--celeba-test-grids', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    jobs = [item.strip() for item in args.jobs.split(',') if item.strip()]
    for name, test_grids in (('svhn', args.svhn_test_grids), ('celeba', args.celeba_test_grids)):
        if name in jobs and test_grids is None:
            parser.error(f'--jobs {name} requires --{name}-test-grids')
    meta = json.loads((args.inputs / 'input.json').read_text(encoding='utf-8'))
    need_cifar = any(name in jobs for name in ('red', 'green', 'blue'))
    pixels = labels = indices = None
    channels = {}
    if need_cifar:
        for name, row in meta['channels'].items():
            channels[name] = {
                'test1000': np.load(row['files']['test1000']['path']),
                'paired100': np.load(row['files']['paired100']['path']),
            }
        with tarfile.open(repo / 'data/cifar-10-python.tar.gz') as archive:
            batch = pickle.load(archive.extractfile('cifar-10-batches-py/data_batch_1'), encoding='bytes')
        labels = np.asarray(batch[b'labels'])
        pixels = np.asarray(batch[b'data']).reshape(-1, 3, 32, 32)
        indices = class_indices(labels, 30)
        assert sorted(class_indices(labels, 1)) == sorted(meta['train_indices'])
    for name in jobs:
        if name in ('red', 'green', 'blue'):
            axis = {'red': 0, 'green': 1, 'blue': 2}[name]
            train_raw = list(pixels[indices, axis])
            test_raws = {
                'paired100': channels[name]['paired100'],
                'test1000': channels[name]['test1000'],
            }
            for kind in ('bin128', 'bin_permedian', '4level'):
                train, tests = prepare_sets(train_raw, test_raws, kind)
                run_suite(name, train, tests, kind, args.out / 'p17-cifar' / name / f'{kind}.json')
        elif name in ('svhn', 'celeba'):
            roots = {
                'svhn': (args.svhn_test_grids,
                         repo / 'results/2026-11-10-elsner-color-modes/svhn/gray/train/transcribed_data_og'),
                'celeba': (args.celeba_test_grids,
                           repo / 'results/2026-11-10-elsner-color-modes/celeba/gray/train/transcribed_data_og'),
            }
            test_root, train_root = roots[name]
            train_raw = load_dats(train_root)
            test_raw = load_dats(test_root)
            for kind in ('bin128', 'bin_permedian', '4level'):
                train, tests = prepare_sets(train_raw, {name: test_raw}, kind)
                run_suite(name, train, tests, kind, args.out / 'p17-other' / name / f'{kind}.json')
        else:
            raise SystemExit(f'unknown job {name}')
    print('BATCH30_P17_DONE', flush=True)


if __name__ == '__main__':
    main()
