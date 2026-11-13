"""MNIST scale comparison using batch 31 masks, merge rules and cost model."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import time

from batch31_mnist import (
    CHECKPOINTS, RAW, SEED, cells_2d, choose_indices, classify, flatten,
    hilbert_order, learn_apply, load_idx, pick, plain_cost, prepare, save, slice_model,
)
from batch30_mask_baselines import alphabet_of, candidate_cost
from lossless_nd_candidate import LosslessNDCandidate
from nd_bpe_incremental_probe import Index
import numpy as np
import psutil


def expanded(images, scale):
    return [np.repeat(np.repeat(image, scale, axis=0), scale, axis=1) for image in images]


def replay_chunk(model, images, cpu, start):
    """Replay the existing encoder's replacement order once per test image."""
    psutil.Process().cpu_affinity([cpu])
    prefixes = {budget: slice_model(model, budget) for budget in CHECKPOINTS}
    clones = {
        budget: LosslessNDCandidate.from_dictionary(json.loads(json.dumps(part.dictionary())))
        for budget, part in prefixes.items()
    }
    at_step = {}
    for budget, part in prefixes.items():
        at_step.setdefault(len(part.rules), []).append(budget)
    rows = {
        budget: dict(sequences=[], failures=[], replacements=Counter())
        for budget in CHECKPOINTS
    }
    shape_kinds = {token: classify(model.shapes[token])[0] for _, token in model.rules}
    base = {value: index for index, value in enumerate(model.base)}
    for image_index, image in enumerate(images):
        shape = image.shape
        cells = cells_2d(image)
        index = Index()
        # The frozen MNIST learning split contains every mask value.
        state = model._state(shape, {point: base[value] for point, value in cells.items()},
                             index, 0, len(model.base))
        replaced = Counter()

        def snapshot(step):
            for budget in at_step.get(step, ()):
                row = rows[budget]
                tokens = [region[0] for region in sorted(state.regions.values(), key=lambda r: r[1])]
                packet = dict(format='nd-bpe-candidate-packet', version=1, shape=list(shape),
                              domain=None, literals=[], tokens=tokens)
                assert sum(replaced.values()) == len(cells) - len(tokens)
                if image_index == 0:
                    assert prefixes[budget].encode(shape, cells) == packet
                try:
                    if clones[budget].decode(json.loads(json.dumps(packet))) != (shape, cells):
                        row['failures'].append(start + image_index)
                except ValueError:
                    row['failures'].append(start + image_index)
                row['sequences'].append(tokens)
                row['replacements'].update(replaced)

        snapshot(0)
        for step, (key, token) in enumerate(model.rules, 1):
            candidates = []
            for _, left, right in index.occurrences.get(key, ()):
                if state.regions[left][1] > state.regions[right][1]:
                    left, right = right, left
                candidates.append((state.edges[left][right], left, right))
            for _, left, right in sorted(candidates):
                replaced[shape_kinds[token]] += state.replace(left, right, token)
            snapshot(step)
        if (image_index + 1) % 100 == 0:
            print('2D_TEST', 'chunk_start', start, image_index + 1, '/', len(images), flush=True)
    return rows


def evaluate_2d(model, images, cpus):
    if len(cpus) == 1:
        return replay_chunk(model, images, cpus[0], 0)
    boundaries = [len(images) * i // len(cpus) for i in range(len(cpus) + 1)]
    with ProcessPoolExecutor(max_workers=len(cpus)) as pool:
        futures = [
            pool.submit(replay_chunk, model, images[start:end], cpu, start)
            for cpu, start, end in zip(cpus, boundaries, boundaries[1:])
        ]
        merged = {budget: dict(sequences=[], failures=[], replacements=Counter()) for budget in CHECKPOINTS}
        for future in futures:
            for budget, rows in future.result().items():
                merged[budget]['sequences'].extend(rows['sequences'])
                merged[budget]['failures'].extend(rows['failures'])
                merged[budget]['replacements'].update(rows['replacements'])
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, choices=(3, 4), required=True)
    parser.add_argument('--mask', choices=('bin128', '4level'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hilbert-only', action='store_true')
    parser.add_argument('--test-workers', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    allowed_cpus = process.cpu_affinity()
    process.cpu_affinity(allowed_cpus[:1])
    started = time.perf_counter()
    files = [RAW / name for name in ('train-images-idx3-ubyte', 't10k-images-idx3-ubyte')]
    checksums = {}
    for path in files:
        with path.open('rb') as handle:
            checksums[path.name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    train, test = map(load_idx, files)
    assert train.shape == (60000, 28, 28) and test.shape == (10000, 28, 28)
    chosen = choose_indices(len(train), 1000)[:300]
    train_images, *_ = prepare(expanded(train[chosen], args.scale), args.mask)
    images, flat, byte_blob, packed_blob = prepare(expanded(test[:1000], args.scale), args.mask)
    del train, test
    side = 28 * args.scale
    alphabet = alphabet_of(args.mask)
    pixels = len(images) * side * side
    cost_inputs = (alphabet, pixels, flat, byte_blob, packed_blob)
    method = 'hilbert' if args.hilbert_only else 'row'
    report = dict(mask=args.mask, side=side, scale=args.scale, seed=SEED, train_indices=chosen,
                  test_count=len(images), pixels=pixels, inputs_sha256=checksums, order=method,
                  cpu_affinity=process.cpu_affinity(), priority=int(process.nice()), results={})
    report_path = args.out / 'report.json'
    order = hilbert_order(side) if args.hilbert_only else list(np.ndindex(side, side))
    print('START', side, args.mask, method, 'available_bytes', psutil.virtual_memory().available, flush=True)
    rules, snapshots = learn_apply([flatten(image, order) for image in train_images],
                                   {'test': [flatten(image, order) for image in images]},
                                   alphabet, max(CHECKPOINTS), set(CHECKPOINTS))
    save(args.out / f'{method}-rules.json', rules)
    rows = report['results'][method] = {}
    for budget in CHECKPOINTS:
        (encoded, bad), step = pick(snapshots['test'], budget)
        row = plain_cost(encoded, step, *cost_inputs)
        row.update(lossless_bad=bad, total_bits=row['bits_per_cell'] * pixels)
        rows[str(budget)] = row
        print(method, budget, row['tokens'], row['bits_per_cell'], len(bad), flush=True)
    save(report_path, report)
    del snapshots, encoded, order
    gc.collect()

    if not args.hilbert_only:
        print('2D_LEARN', side, args.mask, 'available_bytes', psutil.virtual_memory().available, flush=True)
        model = LosslessNDCandidate(2)
        model.learn([((side, side), cells_2d(image)) for image in train_images],
                    budget=max(CHECKPOINTS), score='frequency')
        save(args.out / '2d-dictionary.json', model.dictionary())
        gc.collect()
        print('2D_LEARN_DONE', len(model.rules), flush=True)
        # Keep the learning phase single-core. A fresh pool only replays held-out inputs.
        available_gib = psutil.virtual_memory().available / 2**30
        workers = min(args.test_workers, len(allowed_cpus) - 1, max(1, int(available_gib) - 4))
        test_cpus = allowed_cpus[1:1 + workers] if workers > 1 else allowed_cpus[:1]
        report['test_cpus'] = test_cpus
        print('2D_TEST_WORKERS', len(test_cpus), flush=True)
        observations = evaluate_2d(model, images, test_cpus)
        rows = report['results']['2d'] = {}
        for budget in CHECKPOINTS:
            part, observed = slice_model(model, budget), observations[budget]
            row = candidate_cost(observed['sequences'], part, 0, *cost_inputs)
            # Keep batch 31's estimate and show the cost if signed deltas need more bits.
            deltas = [d for (left, right, delta), token in part.rules for d in delta]
            needed = max((1 + (d if d >= 0 else ~d).bit_length() for d in deltas), default=1)
            extra = max(0, needed - 6) * 2 * len(part.rules)
            total = row['bits_per_cell'] * pixels
            row.update(lossless_bad=observed['failures'], replacements=dict(observed['replacements']),
                       total_bits=total, signed_delta_bits=needed,
                       wider_delta_total_bits=total + extra,
                       wider_delta_bits_per_cell=(total + extra) / pixels)
            assert sum(row['replacements'].values()) == pixels - row['tokens']
            rows[str(budget)] = row
            print('2d', budget, row['tokens'], row['bits_per_cell'], len(row['lossless_bad']), flush=True)
    report['seconds'] = time.perf_counter() - started
    report['complete'] = True
    save(report_path, report)
    assert all(not row['lossless_bad'] for rows in report['results'].values() for row in rows.values())
    print('DONE', side, args.mask, round(report['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
