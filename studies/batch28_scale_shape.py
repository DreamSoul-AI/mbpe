"""Batch 28: larger candidate tests, other datasets, rank and masks.

Uses the existing candidate as-is. Does not change its default learn path.
Official repositories are not edited.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMBA_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

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


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def h0(values):
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def id_bits(size):
    return max(1, math.ceil(math.log2(max(size, 2))))


def cells_2d(image):
    return {(int(y), int(x)): int(image[y, x]) for y in range(image.shape[0]) for x in range(image.shape[1])}


def load_dats(root):
    files = sorted(root.rglob('*.dat'), key=lambda path: int(path.stem))
    arrays = []
    for path in files:
        grid = torch.load(path, map_location='cpu', weights_only=False).long().numpy()
        arrays.append(grid)
    return arrays


def class_indices(labels, per_class):
    chosen = []
    for label in range(10):
        found = np.flatnonzero(labels == label)[:per_class]
        if len(found) < per_class:
            raise RuntimeError(f'class {label} has {len(found)}')
        chosen.extend(int(i) for i in found)
    return chosen


def learn_and_test(rank, train, test, budget):
    model = LosslessNDCandidate(rank)
    started = time.perf_counter()
    train_lengths = model.learn(train, budget=budget, score='frequency')
    learn_seconds = time.perf_counter() - started
    clone = LosslessNDCandidate.from_dictionary(json.loads(json.dumps(model.dictionary())))
    lengths = []
    tokens = []
    literal_count = 0
    bad = []
    for index, (shape, cells) in enumerate(test):
        packet = model.encode(shape, cells)
        restored = clone.decode(json.loads(json.dumps(packet)))
        if restored != (tuple(shape), cells):
            bad.append(index)
        lengths.append(len(packet['tokens']))
        tokens.extend(packet['tokens'])
        literal_count += len(packet['literals'])
    domain = len(test[0][1]) if test else 0
    return {
        'budget': budget,
        'actual_rules': len(model.rules),
        'train_images': len(train),
        'test_images': len(test),
        'domain_cells': domain,
        'train_ratio': sum(train_lengths) / (len(train) * domain) if train else None,
        'test_ratio': sum(lengths) / (len(test) * domain) if test else None,
        'test_tokens': sum(lengths),
        'lossless_bad': bad,
        'learn_seconds': learn_seconds,
        'token_bits': h0(tokens) * len(tokens),
        'dictionary': model.dictionary(),
        'literal_count': literal_count,
        'n_tokens': len(tokens),
    }


def bit_row(result):
    dictionary = result['dictionary']
    vocab = len(dictionary['base']) + len(dictionary['rules'])
    width = id_bits(vocab)
    rank = dictionary['rank']
    dict_bits = 256 + len(dictionary['rules']) * (2 * width + rank * 6)
    images = result['test_images']
    shape_bits = images * (8 * rank)
    domain_bits = images
    literal_bits = result['literal_count'] * (width + 8)
    total = result['token_bits'] + dict_bits + shape_bits + domain_bits + literal_bits
    pixels = images * result['domain_cells']
    return {'test_ratio': result['test_ratio'], 'tokens': result['test_tokens'],
            'actual_rules': result['actual_rules'], 'token_bits': result['token_bits'],
            'dictionary_bits': dict_bits, 'shape_bits': shape_bits,
            'domain_flag_bits': domain_bits, 'literal_bits': literal_bits,
            'total_bits': total, 'bits_per_pixel': total / pixels if pixels else None,
            'lossless_bad': len(result['lossless_bad'])}


def bpe1d(train_seqs, test_seqs, budget):
    sequences = [list(seq) for seq in train_seqs]
    rules = []
    shapes = {value: [value] for value in range(256)}
    next_id = 256
    for _ in range(budget):
        counts = Counter()
        for seq in sequences:
            counts.update(zip(seq, seq[1:]))
        if not counts:
            break
        pair, gain = min(counts.items(), key=lambda item: (-item[1], item[0]))
        if gain < 2:
            break
        rules.append((pair, next_id))
        merged = []
        for seq in sequences:
            out = []
            index = 0
            while index < len(seq):
                if index + 1 < len(seq) and (seq[index], seq[index + 1]) == pair:
                    out.append(next_id)
                    index += 2
                else:
                    out.append(seq[index])
                    index += 1
            merged.append(out)
        sequences = merged
        next_id += 1
    shapes_apply = {value: [value] for value in range(256)}
    for (left, right), token in rules:
        shapes_apply[token] = shapes_apply[left] + shapes_apply[right]
    bad = []
    lengths = []
    for index, seq in enumerate(test_seqs):
        current = list(seq)
        for (left, right), token in rules:
            out = []
            cursor = 0
            while cursor < len(current):
                if cursor + 1 < len(current) and (current[cursor], current[cursor + 1]) == (left, right):
                    out.append(token)
                    cursor += 2
                else:
                    out.append(current[cursor])
                    cursor += 1
            current = out
        flat = []
        for token in current:
            flat.extend(shapes_apply[token])
        if flat != list(seq):
            bad.append(index)
        lengths.append(len(current))
    width = len(test_seqs[0])
    return {'actual_rules': len(rules), 'test_ratio': sum(lengths) / (len(test_seqs) * width),
            'test_tokens': sum(lengths), 'lossless_bad': bad}


def circle_cells(image, radius=12):
    cells = {}
    for y in range(32):
        for x in range(32):
            if (y - 15.5) ** 2 + (x - 15.5) ** 2 <= radius ** 2:
                cells[(y, x)] = int(image[y, x])
    return cells


def flatten(image):
    values = [int(value) for value in image.reshape(-1)]
    return (1024,), {(index,): value for index, value in enumerate(values)}


def volume(rgb):
    cells = {(y, x, channel): int(rgb[channel, y, x])
             for channel in range(3) for y in range(32) for x in range(32)}
    return (32, 32, 3), cells


def run_p8(out, labels, pixels, frozen):
    rows = []
    for count, per_class in ((10, 1), (300, 30)):
        indices = class_indices(labels, per_class)
        if count == 10:
            assert sorted(indices) == sorted(frozen['train_indices'])
        for channel, axis in (('red', 0), ('green', 1), ('blue', 2)):
            if count == 10:
                train_images = frozen['channels'][channel]['train']
            else:
                train_images = pixels[indices, axis]
            test_images = frozen['channels'][channel]['test1000']
            train = [((32, 32), cells_2d(image)) for image in train_images]
            test = [((32, 32), cells_2d(image)) for image in test_images]
            result = learn_and_test(2, train, test, 256)
            result.pop('dictionary')
            row = {'channel': channel, 'train_images': count, **{k: v for k, v in result.items()
                   if k not in ('token_bits', 'literal_count', 'n_tokens')}}
            rows.append(row)
            save(out / 'report.json', rows)
            print('P8', channel, count, row['test_ratio'], row['lossless_bad'], flush=True)
    return rows


def run_p9(out, repo, svhn_test_grids, celeba_test_grids):
    specs = [
        ('svhn', svhn_test_grids,
         repo / 'results/2026-11-10-elsner-color-modes/svhn/gray/train/transcribed_data_og',
         92.7568359375, 100, 100),
        ('celeba', celeba_test_grids,
         repo / 'results/2026-11-10-elsner-color-modes/celeba/gray/train/transcribed_data_og',
         95.7356770833, 90, 88),
    ]
    rows = []
    for name, test_root, train_root, elsner_percent, expect_test, elsner_native in specs:
        test_arrays = load_dats(test_root)
        train_arrays = load_dats(train_root)
        assert len(test_arrays) == expect_test
        assert len(train_arrays) == 10
        assert all(array.shape == (32, 32) for array in test_arrays + train_arrays)
        result = learn_and_test(2, [((32, 32), cells_2d(image)) for image in train_arrays],
                                [((32, 32), cells_2d(image)) for image in test_arrays], 256)
        result.pop('dictionary')
        row = {'dataset': name, 'elsner_percent': elsner_percent,
               'elsner_native_ok': f'{elsner_native}/{expect_test}',
               **{k: v for k, v in result.items() if k not in ('token_bits', 'literal_count', 'n_tokens')}}
        rows.append(row)
        save(out / 'report.json', rows)
        print('P9', name, row['test_ratio'], len(row['lossless_bad']), flush=True)
    return rows


def run_p10(out, labels, pixels, frozen):
    indices = class_indices(labels, 30)
    paired = frozen['channels']['red']['paired_positions']
    report = {}
    train_flat = []
    test_flat = []
    train_seq = []
    test_seq = []
    for index in indices:
        image = pixels[index, 0]
        train_flat.append(flatten(image))
        train_seq.append([value for _, value in sorted(train_flat[-1][1].items())])
    for image in frozen['channels']['red']['paired100']:
        test_flat.append(flatten(image))
        test_seq.append([value for _, value in sorted(test_flat[-1][1].items())])
    candidate = learn_and_test(1, train_flat, test_flat, 256)
    candidate.pop('dictionary')
    plain = bpe1d(train_seq, test_seq, 256)
    report['oned'] = {'channel': 'red', 'train_images': 300, 'test_images': 100,
                      'candidate': {k: v for k, v in candidate.items()
                                    if k not in ('token_bits', 'literal_count', 'n_tokens')},
                      'plain_bpe': plain}
    save(out / 'report.json', report)
    print('P10-1D', candidate['test_ratio'], plain['test_ratio'], flush=True)

    budgets = {'red': 86, 'green': 85, 'blue': 85}
    channel_rows = []
    total_tokens = 0
    total_bad = 0
    total_rules = 0
    for channel, axis in (('red', 0), ('green', 1), ('blue', 2)):
        train = [((32, 32), cells_2d(image)) for image in pixels[indices, axis]]
        test = [((32, 32), cells_2d(image)) for image in frozen['channels'][channel]['paired100']]
        result = learn_and_test(2, train, test, budgets[channel])
        result.pop('dictionary')
        total_tokens += result['test_tokens']
        total_bad += len(result['lossless_bad'])
        total_rules += result['actual_rules']
        channel_rows.append({'channel': channel, **{k: v for k, v in result.items()
                            if k not in ('token_bits', 'literal_count', 'n_tokens')}})
    train_vol = [volume(pixels[index]) for index in indices]
    test_vol = []
    for position in range(100):
        stacked = np.stack([frozen['channels'][name]['paired100'][position] for name in ('red', 'green', 'blue')])
        test_vol.append(volume(stacked))
    volume_result = learn_and_test(3, train_vol, test_vol, 256)
    volume_result.pop('dictionary')
    report['threed'] = {
        'total_budget': 256,
        'per_channel_budgets': budgets,
        'per_channel': channel_rows,
        'per_channel_total_tokens': total_tokens,
        'per_channel_ratio': total_tokens / (100 * 3072),
        'per_channel_bad': total_bad,
        'per_channel_rules': total_rules,
        'volume': {k: v for k, v in volume_result.items()
                   if k not in ('token_bits', 'literal_count', 'n_tokens')},
    }
    save(out / 'report.json', report)
    print('P10-3D', report['threed']['per_channel_ratio'], volume_result['test_ratio'], flush=True)

    train_mask = [((32, 32), circle_cells(pixels[index, 0])) for index in indices]
    test_mask = [((32, 32), circle_cells(image)) for image in frozen['channels']['red']['paired100']]
    masked = learn_and_test(2, train_mask, test_mask, 256)
    masked.pop('dictionary')
    report['mask'] = {'mask': 'circle center (15.5,15.5) radius 12, red channel',
                      'paired_positions_checked': paired[:3],
                      **{k: v for k, v in masked.items() if k not in ('token_bits', 'literal_count', 'n_tokens')}}
    save(out / 'report.json', report)
    print('P10-mask', masked['domain_cells'], masked['test_ratio'], masked['lossless_bad'], flush=True)
    return report


def run_p11(out, labels, pixels, frozen):
    indices = class_indices(labels, 30)
    rows = []
    for budget in (512, 1024):
        for channel, axis in (('red', 0), ('green', 1), ('blue', 2)):
            train = [((32, 32), cells_2d(image)) for image in pixels[indices, axis]]
            test_images = frozen['channels'][channel]['paired100']
            test = [((32, 32), cells_2d(image)) for image in test_images]
            result = learn_and_test(2, train, test, budget)
            raw = test_images.astype(np.uint8).tobytes()
            cost = bit_row(result)
            cost.update({'channel': channel, 'budget': budget, 'train_images': 300, 'test_images': 100,
                         'raw_bits': 100 * 1024 * 8,
                         'raw_zero_order_bits': h0(raw) * len(raw),
                         'zlib9_bits': len(zlib.compress(raw, 9)) * 8,
                         'lossless_bad_indices': result['lossless_bad']})
            rows.append(cost)
            save(out / 'report.json', rows)
            print('P11', channel, budget, cost['test_ratio'], cost['bits_per_pixel'], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--svhn-test-grids', type=Path, required=True)
    parser.add_argument('--celeba-test-grids', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    meta = json.loads((args.inputs / 'input.json').read_text(encoding='utf-8'))
    frozen = {'train_indices': meta['train_indices'],
              'channels': {}}
    for name, row in meta['channels'].items():
        frozen['channels'][name] = {
            'train': np.load(row['files']['train']['path']),
            'test1000': np.load(row['files']['test1000']['path']),
            'paired100': np.load(row['files']['paired100']['path']),
            'paired_positions': meta['paired_positions_in_1000'],
        }
    with tarfile.open(repo / 'data/cifar-10-python.tar.gz') as archive:
        batch = pickle.load(archive.extractfile('cifar-10-batches-py/data_batch_1'), encoding='bytes')
    labels = np.asarray(batch[b'labels'])
    pixels = np.asarray(batch[b'data']).reshape(-1, 3, 32, 32)
    run_p8(args.out / 'p8-cifar1000', labels, pixels, frozen)
    run_p9(args.out / 'p9-svhn-celeba', repo, args.svhn_test_grids, args.celeba_test_grids)
    run_p10(args.out / 'p10-shapes', labels, pixels, frozen)
    run_p11(args.out / 'p11-budget', labels, pixels, frozen)
    print('BATCH28_DONE', flush=True)


if __name__ == '__main__':
    main()
