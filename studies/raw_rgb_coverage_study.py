"""Compare raw RGB BPE coverage, budget and parsing on frozen CPU inputs."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import threading

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ.update(CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false')

import numpy as np
import psutil
from tokenizers import Tokenizer, models, trainers
from rgb_capacity_study import digest_file, read_json, resource_receipt, save_json, select_indices, source_images
from rgb_large_vocab_study import expand, texts
from color_template_codec import color_basis, inverse_color_basis
from rgb_input_codec import difference, restore_difference
from rgb_linear_codec import scan_points

BUDGETS = (16384, 65536, 150000, 200000)
COUNTS = (300, 1000, 3000)
DATASETS = ('cifar', 'svhn')


def study_grid(args):
    if args.scan != 'column':
        return (6000,), (600000,)
    if args.only_6000:
        return (6000,), (200000, 400000, 600000)
    if args.extend:
        return (3000, 6000), (200000, 400000, 600000)
    return ((1000, 3000) if args.fixed_test else COUNTS), BUDGETS


def spatial_order(transform, order):
    if order is not None:
        return order
    return 0 if transform == 'raw' else 1


def channel_basis(image, transform):
    if transform == 'color':
        return color_basis(image)
    if transform in ('chain1', 'chain2'):
        red, green, blue = image
        green_delta = np.subtract(green, red, dtype=np.uint8)
        blue_delta = np.subtract(blue, green, dtype=np.uint8)
        if transform == 'chain2':
            blue_delta = np.subtract(blue_delta, green_delta, dtype=np.uint8)
        return np.stack((red, green_delta, blue_delta))
    return image


def restore_basis(image, transform):
    if transform == 'color':
        return inverse_color_basis(image)
    if transform in ('chain1', 'chain2'):
        red, green_delta, blue_delta = image
        green = np.add(red, green_delta, dtype=np.uint8)
        if transform == 'chain2':
            blue_delta = np.add(blue_delta, green_delta, dtype=np.uint8)
        blue = np.add(green, blue_delta, dtype=np.uint8)
        return np.stack((red, green, blue))
    return image


def difference_axes(order, direction):
    if direction == 'mixed':
        if order != 2:
            raise ValueError('mixed difference uses one pass per spatial axis')
        return (-1, -2)
    axis = -2 if direction == 'vertical' else -1
    return (axis,) * order


def apply_spatial_pass(image, axis, operation):
    if axis == -1:
        return operation(image)
    return operation(image.swapaxes(-1, -2)).swapaxes(-1, -2)


def input_strings(image, transform, scan='column', order=None, direction='horizontal'):
    image = channel_basis(image, transform)
    for axis in difference_axes(spatial_order(transform, order), direction):
        image = apply_spatial_pass(image, axis, difference)
    if scan == 'column':
        return texts(image)
    ys, xs = zip(*scan_points(scan))
    return [''.join(chr(256 + int(v)) for v in channel) for channel in image[:, ys, xs]]


def load_arrays(path, identity):
    if digest_file(path) != identity['array_sha256']:
        raise ValueError(f'array changed: {path}')
    with np.load(path, allow_pickle=False) as packet:
        images = packet['images']
    for image, row in zip(images, identity['samples'], strict=True):
        if sha256(image.tobytes()).hexdigest() != row['sha256']:
            raise ValueError('pixel identity mismatch')
    return images


def freeze(args):
    prior = read_json(args.prior / 'inputs.json')['datasets']
    shared = read_json(args.shared / 'inputs.json')['datasets']
    old = [read_json(path / 'inputs.json')['datasets'] for path in (args.h1, args.h2)]
    frozen_test = read_json(args.fixed_test / 'inputs.json')['datasets'] if args.fixed_test else None
    extension = read_json(args.extend / 'inputs.json')['datasets'] if args.extend else None
    output = {}
    for dataset in DATASETS:
        target = args.out / dataset
        target.mkdir(exist_ok=True)
        learn = prior[dataset]['splits']['learn']
        test = old[1][dataset]
        records = [*prior[dataset]['splits'].values(), *shared[dataset]['splits'].values(),
                   *(manifest[dataset] for manifest in old)]
        seen = {row['sha256'] for part in records for row in part['samples']}
        blocked = {name: set(indices) for name, indices in shared[dataset]['excluded_indices'].items()}
        for part in records:
            blocked.setdefault(part['source'], set()).update(row['index'] for row in part['samples'])
        train_images, train_labels = source_images(dataset, args.backup / learn['source'], learn['source_sha256'])
        test_images, test_labels = source_images(dataset, args.backup / test['source'], test['source_sha256'])
        for images, identity in ((train_images, learn), (test_images, test)):
            seen.update(sha256(images[i].tobytes()).hexdigest() for i in blocked[identity['source']])
        if extension:
            previous = extension[dataset]
            for group in ('3000', 'test'):
                part = previous[group]
                blocked[part['source']].update(row['index'] for row in part['samples'])
                seen.update(row['sha256'] for row in part['samples'])
            existing = previous['3000']['samples']
            if frozen_test:
                sampled = frozen_test[dataset]['test']['samples']
                seen.update(row['sha256'] for row in sampled)
            extra = select_indices(train_labels, train_images, 3000, args.seed, blocked[learn['source']], seen)
            pool = existing + extra
            if not frozen_test:
                sampled = select_indices(test_labels, test_images, 500, 171124, blocked[test['source']], seen)
        elif frozen_test:
            sampled = frozen_test[dataset]['test']['samples']
            seen.update(row['sha256'] for row in sampled)
            pool = select_indices(train_labels, train_images, 3000, args.seed, blocked[learn['source']], seen)
            existing = []
            for label in range(10):
                members = [row for row in pool if row['class'] == label]
                existing.extend(members[:100])
        else:
            extra = select_indices(train_labels, train_images, 2000, args.seed + 1, blocked[learn['source']], seen)
            sampled = select_indices(test_labels, test_images, 500, 171122, blocked[test['source']], seen)
            existing = learn['samples']
            pool = existing + extra
        if extension:
            groups = [('3000', existing), ('6000', pool)]
        else:
            groups = [('1000', existing), ('3000', pool)]
            if not frozen_test:
                rng = np.random.default_rng(args.seed)
                small = []
                for label in range(10):
                    members = [row for row in existing if row['class'] == label]
                    small.extend(members[i] for i in rng.permutation(len(members))[:30])
                groups.insert(0, ('300', small))
        output[dataset] = {}
        splits = [(name, rows, train_images, learn) for name, rows in groups]
        splits.append(('test', sampled, test_images, test))
        for name, rows, images, source in splits:
            path = target / f'{name}.npz'
            with path.open('xb') as handle:
                np.savez(handle, images=np.stack([images[row['index']] for row in rows]))
            output[dataset][name] = {**source, 'samples': rows, 'array_sha256': digest_file(path)}
        print('frozen', dataset, flush=True)
    save_json(args.out / 'inputs.json', {'datasets': output, 'seed': args.seed, 'test_seed': 171124 if extension else 171122,
                                      'resource': resource_receipt()})


def fit(args):
    directory = args.out / args.dataset
    source = args.fixed_inputs or args.out
    identity = read_json(source / 'inputs.json')['datasets'][args.dataset][str(args.count)]
    images = load_arrays(source / args.dataset / f'{args.count}.npz', identity)
    _, budgets = study_grid(args)
    alphabet = [chr(256 + i) for i in range(256)]
    enc = Tokenizer(models.BPE())
    strings = (text for image in images
               for text in input_strings(image, args.transform, args.scan,
                                         args.spatial_order, args.spatial_direction))
    trainer = trainers.BpeTrainer(vocab_size=budgets[-1] + 256, min_frequency=1,
                                  initial_alphabet=alphabet, show_progress=False)
    enc.train_from_iterator(strings, length=len(images) * 3, trainer=trainer)
    full = json.loads(enc.to_str())['model']
    models_receipt = {}
    for budget in budgets:
        pairs = full['merges'][:budget]
        vocab = {atom: full['vocab'][atom] for atom in alphabet}
        vocab.update({a + b: full['vocab'][a + b] for a, b in pairs})
        path = directory / f'{args.count}-{budget}.json'
        Tokenizer(models.BPE(vocab=vocab, merges=[tuple(pair) for pair in pairs])).save(str(path))
        if args.transform == 'raw' and args.extend and args.count == 3000 and budget == 200000:
            parent = args.extend / args.dataset / '3000-200000.json'
            if read_json(path)['model'] != read_json(parent)['model']:
                raise ValueError('extended fit changed the existing 200000-rule prefix')
        models_receipt[str(budget)] = {'path': path.name, 'sha256': digest_file(path), 'rules': len(pairs)}
    if args.count == 1000 and not args.fixed_test:
        prior = read_json(args.models / args.dataset / 'raw-200000.json')['model']
        if full != prior:
            raise ValueError('reconstructed 1000-image dictionary differs from frozen original')
    save_json(directory / f'{args.count}-fit.json', {'models': models_receipt, 'resource': resource_receipt()})


def restore_stream(ids, words, rows, transform='raw', scan='column', order=None, direction='horizontal'):
    lengths = {i: len(w) for i, w in words.items()}
    cursor = 0
    for row in rows:
        channels = []
        for _ in range(3):
            start = cursor
            cells = 0
            while cells < 1024:
                if cursor == len(ids):
                    raise ValueError('incomplete channel')
                cells += lengths[ids[cursor]]
                cursor += 1
            if cells != 1024:
                raise ValueError('token crosses channel boundary')
            channel = expand(ids[start:cursor], words)
            if scan != 'column':
                ys, xs = zip(*scan_points(scan))
                moved = np.empty_like(channel)
                moved[ys, xs] = channel.T.ravel()
                channel = moved
            channels.append(channel)
        image = np.stack(channels)
        for axis in reversed(difference_axes(spatial_order(transform, order), direction)):
            image = apply_spatial_pass(image, axis, restore_difference)
        image = restore_basis(image, transform)
        if sha256(image.tobytes()).hexdigest() != row['sha256']:
            raise ValueError('independent decode differs')
    if cursor != len(ids):
        raise ValueError('trailing tokens')


def seal(args):
    counts, _ = study_grid(args)
    if args.extend and args.fixed_test:
        counts = (6000,)
    dictionaries = {d: {str(n): read_json(args.out / d / f'{n}-fit.json')['models']
                        for n in counts} for d in DATASETS}
    source = args.fixed_inputs or args.out
    save_json(args.out / 'seal.json', {'inputs': digest_file(source / 'inputs.json'), 'dictionaries': dictionaries,
                                      'transform': args.transform, 'scan': args.scan,
                                      'spatial_order': spatial_order(args.transform, args.spatial_order),
                                      'spatial_direction': args.spatial_direction})


def conditions(args):
    if args.count == 0:
        identity = read_json(args.h2 / 'inputs.json')['datasets'][args.dataset]
        source = args.models / args.dataset
        models_receipt = read_json(args.models / 'sealed.json')['models']
        entries = []
        for budget in (150000, 200000):
            name = f'raw-{budget}.json'
            entries.append((budget, source / name, models_receipt[str(Path(args.dataset) / name)]['sha256'], budget))
        return identity, args.h2 / args.dataset / 'test.npz', entries
    source = args.fixed_inputs or args.out
    manifest = source / 'inputs.json'
    frozen = read_json(args.out / 'seal.json')
    if frozen.get('transform', 'raw') != args.transform:
        raise ValueError('representation differs from seal')
    if frozen.get('scan', 'column') != args.scan:
        raise ValueError('scan differs from seal')
    if spatial_order(args.transform, frozen.get('spatial_order')) != spatial_order(args.transform, args.spatial_order):
        raise ValueError('spatial order differs from seal')
    if frozen.get('spatial_direction', 'horizontal') != args.spatial_direction:
        raise ValueError('spatial direction differs from seal')
    if digest_file(manifest) != frozen['inputs']:
        raise ValueError('holdout identity changed')
    identity = read_json(manifest)['datasets'][args.dataset]['test']
    directory = args.out / args.dataset
    entries = [(int(budget), directory / row['path'], row['sha256'], row['rules'])
               for budget, row in frozen['dictionaries'][args.dataset][str(args.count)].items()]
    return identity, source / args.dataset / 'test.npz', entries


def evaluate(args):
    identity, path, dictionaries = conditions(args)
    directory = args.out / args.dataset
    images = load_arrays(path, identity)
    strings = [input_strings(image, args.transform, args.scan,
                             args.spatial_order, args.spatial_direction) for image in images]
    result = {}
    for budget, dictionary, digest, rules in dictionaries:
        if digest_file(dictionary) != digest:
            raise ValueError('sealed dictionary changed')
        bpe = Tokenizer.from_file(str(dictionary))
        words = {i: w for w, i in bpe.get_vocab().items()}
        minimum = Tokenizer(models.Unigram([(words[i], -1.0) for i in range(len(words))]))
        if minimum.get_vocab() != bpe.get_vocab():
            raise ValueError('minimum parser changed dictionary IDs')
        counts = {}
        for name, tokenizer in [('bpe', bpe), ('minimum', minimum)]:
            flat, lengths = [], []
            for channels in strings:
                ids = [i for channel in channels for i in tokenizer.encode(channel, add_special_tokens=False).ids]
                lengths.append(len(ids))
                flat.extend(ids)
            packet = directory / f'{args.count}-{budget}-{name}.npy'
            with packet.open('xb') as handle:
                np.save(handle, np.asarray(flat, dtype=np.uint32), allow_pickle=False)
            counts[name] = {'lengths': lengths, 'packet': packet.name, 'sha256': digest_file(packet)}
        if any(a > b for a, b in zip(counts['minimum']['lengths'], counts['bpe']['lengths'], strict=True)):
            raise ValueError('minimum parse exceeds BPE')
        if args.count == 0:
            old = read_json(args.models / args.dataset / 'report.json')['results'][f'raw-{budget}-bpe']
            if counts['bpe']['lengths'] != old['per_image_tokens']:
                raise ValueError('H2 BPE baseline differs')
        result[str(budget)] = {'rules': rules, 'parsers': counts}
        print(args.dataset, args.count, budget, [np.mean(row['lengths']) for row in counts.values()], flush=True)
    save_json(directory / f'{args.count}-eval.json', {'budgets': result, 'resource': resource_receipt()})


def decode(args):
    identity, _, dictionaries = conditions(args)
    directory = args.out / args.dataset
    report = read_json(directory / f'{args.count}-eval.json')['budgets']
    checked = {}
    for budget, dictionary, digest, _ in dictionaries:
        if digest_file(dictionary) != digest:
            raise ValueError('decode dictionary changed')
        words = {i: w for w, i in read_json(dictionary)['model']['vocab'].items()}
        for name, receipt in report[str(budget)]['parsers'].items():
            packet = directory / receipt['packet']
            if digest_file(packet) != receipt['sha256']:
                raise ValueError('payload changed')
            restore_stream(np.load(packet, allow_pickle=False).tolist(), words, identity['samples'],
                           args.transform, args.scan, args.spatial_order, args.spatial_direction)
            checked[f'{budget}-{name}'] = len(identity['samples'])
    save_json(directory / f'{args.count}-decode.json', {'checked': checked, 'resource': resource_receipt()})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage', choices=('freeze', 'fit', 'seal', 'evaluate', 'decode'))
    for name in ('out', 'prior', 'shared', 'h1', 'h2', 'backup', 'models'):
        cli.add_argument(f'--{name}', type=Path, required=True)
    cli.add_argument('--dataset', choices=DATASETS)
    cli.add_argument('--count', type=int, choices=(0, *COUNTS, 6000))
    cli.add_argument('--seed', type=int, default=171117)
    cli.add_argument('--fixed-test', type=Path)
    cli.add_argument('--extend', type=Path)
    cli.add_argument('--fixed-inputs', type=Path, help='use an already frozen study manifest and original arrays')
    cli.add_argument('--transform', choices=('raw', 'difference', 'color', 'chain1', 'chain2'), default='raw')
    cli.add_argument('--spatial-order', type=int, choices=range(4), help='number of reversible spatial difference passes')
    cli.add_argument('--spatial-direction', choices=('horizontal', 'vertical', 'mixed'), default='horizontal')
    cli.add_argument('--only-6000', action='store_true', help='evaluate only the frozen 6000-image learning condition')
    cli.add_argument('--scan', choices=('column', 'row', 'hilbert'), default='column')
    args = cli.parse_args()
    if args.only_6000 and (not args.fixed_inputs or args.count not in (None, 6000)):
        cli.error('6000-image comparison requires fixed inputs and count 6000')
    if args.spatial_order is not None and not args.only_6000:
        cli.error('explicit difference orders use the fixed 6000-image comparison')
    if args.spatial_direction != 'horizontal' and not args.only_6000:
        cli.error('direction comparison uses the fixed 6000-image comparison')
    if args.spatial_direction == 'mixed' and args.spatial_order != 2:
        cli.error('mixed difference requires spatial order 2')
    if args.transform != 'raw' and (not args.extend or not args.fixed_inputs or args.count == 0):
        cli.error('transform comparison requires fixed extension inputs')
    if args.scan != 'column' and (args.transform != 'color' or args.count not in (None, 6000)):
        cli.error('scan comparison uses the fixed 6000-image color condition')
    process = psutil.Process()
    process.nice(psutil.IDLE_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:1])

    def check_memory():
        if process.memory_info().rss > 8 * 2**30 or psutil.virtual_memory().available < 4 * 2**30:
            save_json(args.out / f'memory-stop-{os.getpid()}.json', resource_receipt())
            os._exit(75)

    check_memory()
    stop = threading.Event()
    def watch_memory():
        while not stop.wait(0.5):
            check_memory()
    threading.Thread(target=watch_memory, daemon=True).start()
    try:
        globals()[args.stage](args)
    finally:
        stop.set()


if __name__ == '__main__':
    main()
