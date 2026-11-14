"""Frozen independent RGB E1 study; CPU only, no neural-network training.

Stages: freeze -> fit (four jobs) -> evaluate validation -> seal -> evaluate test
-> decode (independent processes) -> summarize. Smoke uses its own model tag
and a balanced validation subset, never the formal test set.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle
import shutil
import struct
import sys
import threading
import time
import os

import numpy as np
import psutil

from lossless_nd_candidate import LosslessNDCandidate
import rgb_input_codec as codec


SOURCES = {
    'cifar': ('cifar-10-batches-py/data_batch_5', 'cifar-10-batches-py/test_batch'),
    'svhn': ('train_32x32.mat', 'test_32x32.mat'),
}
SIZES = {'learn': 1000, 'validation': 200, 'test': 1000}
SEEDS = {'learn': 111401, 'validation': 111402, 'test': 111403}
METHODS = ('raw', 'difference', 'raw-bpe', 'difference-bpe')
CAPACITIES = (2, 3, 192)


def digest_file(path):
    digest = sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, data):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def log(event, **data):
    print(json.dumps({'event': event, **data}), flush=True)


def source_images(dataset, path, expected):
    if digest_file(path) != expected:
        raise ValueError(f'source hash differs: {path}')
    if dataset == 'cifar':
        with path.open('rb') as handle:
            source = pickle.load(handle, encoding='bytes')
        images = source[b'data'].reshape(-1, 3, 32, 32)
        labels = np.asarray(source[b'labels'])
    else:
        from scipy.io import loadmat

        source = loadmat(path, variable_names=('X', 'y'))
        images = np.transpose(source['X'], (3, 2, 0, 1))
        labels = source['y'].ravel() % 10
    if images.shape != (len(labels), 3, 32, 32) or images.dtype != np.uint8:
        raise ValueError('unexpected original RGB source')
    return images, labels


def select_indices(labels, images, count, seed, excluded, seen):
    if count % 10:
        raise ValueError('balanced selection requires a multiple of ten')
    rng = np.random.Generator(np.random.PCG64(seed))
    rows = []
    for label in range(10):
        selected = 0
        for index in rng.permutation(np.flatnonzero(labels == label)):
            index = int(index)
            if index in excluded:
                continue
            digest = sha256(images[index].tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            rows.append({'index': index, 'class': label, 'sha256': digest})
            selected += 1
            if selected == count // 10:
                break
        if selected != count // 10:
            raise ValueError('not enough new unique images; explicit protocol amendment required')
    return rows


def freeze(args):
    args.out.mkdir(parents=True, exist_ok=False)
    backup = read_json(args.backup / 'manifest.json')
    entries = {row['path']: row['sha256'] for row in backup['data']['entries']}
    old = read_json(args.old / 'inputs.json')
    protocol = {
        'version': 1, 'shape': [3, 32, 32], 'sizes': SIZES, 'seeds': SEEDS,
        'rng': 'numpy.PCG64', 'numpy': np.__version__, 'rules': 256,
        'base': list(range(256)), 'methods': METHODS, 'capacities': CAPACITIES,
        'width': 192, 'dtype': 'BF16', 'bits_per_token': 9,
        'per_channel_count_bytes': 2, 'channel_order': 'RGB planar',
        'shared_state': ['shape', 'transform', 'grammar', 'capacity'],
        'old_manifest_sha256': digest_file(args.old / 'inputs.json'),
        'purpose': 'E1 independent RGB; validation for smoke, test withheld until seal',
    }
    source_dir = args.out / 'source'
    source_dir.mkdir()
    protocol['source_sha256'] = {}
    for source in sorted(Path(__file__).parent.glob('*.py')):
        shutil.copyfile(source, source_dir / source.name)
        protocol['source_sha256'][source.name] = digest_file(source)
    save_json(args.out / 'protocol.json', protocol)
    manifest = {'datasets': {}}
    for dataset, names in SOURCES.items():
        directory = args.out / dataset
        directory.mkdir()
        seen, splits, exclusions = set(), {}, {}
        # Seed duplicate exclusions with old RGB content from both sources.
        for name in names:
            relative = 'data/' + name
            images, labels = source_images(dataset, args.backup / relative, entries[relative])
            blocked = {row['index'] for part in old[dataset].values() if part['source'] == relative
                       for row in part['samples']}
            exclusions[relative] = sorted(blocked)
            for index in blocked:
                seen.add(sha256(images[index].tobytes()).hexdigest())
            del images, labels
        for name, split_names in ((names[0], ('learn', 'validation')), (names[1], ('test',))):
            relative = 'data/' + name
            images, labels = source_images(dataset, args.backup / relative, entries[relative])
            for split in split_names:
                rows = select_indices(labels, images, SIZES[split], SEEDS[split],
                                      set(exclusions[relative]), seen)
                selected = np.stack([images[row['index']] for row in rows])
                path = directory / (split + '.npz')
                with path.open('xb') as handle:
                    np.savez(handle, images=selected)
                splits[split] = {'source': relative, 'source_sha256': entries[relative],
                                 'samples': rows, 'array_sha256': digest_file(path)}
                log('frozen', dataset=dataset, split=split, images=len(rows))
            del images, labels
        manifest['datasets'][dataset] = {'splits': splits, 'excluded_indices': exclusions}
    save_json(args.out / 'inputs.json', manifest)


def balanced_subset(rows, limit):
    if limit is None:
        return list(range(len(rows)))
    if limit <= 0 or limit % 10 or limit > len(rows):
        raise ValueError('balanced limit must be a positive multiple of ten within split')
    indices = []
    for label in range(10):
        indices.extend([i for i, row in enumerate(rows) if row['class'] == label][:limit // 10])
    if len(indices) != limit:
        raise ValueError('insufficient balanced rows')
    return indices


def load_images(args, split, limit=None):
    manifest = read_json(args.out / 'inputs.json')['datasets'][args.dataset]['splits'][split]
    path = args.out / args.dataset / (split + '.npz')
    if digest_file(path) != manifest['array_sha256']:
        raise ValueError('frozen array changed')
    indices = balanced_subset(manifest['samples'], limit)
    with np.load(path, allow_pickle=False) as archive:
        images = archive['images'][indices]
    rows = [manifest['samples'][i] for i in indices]
    if any(sha256(image.tobytes()).hexdigest() != row['sha256']
           for image, row in zip(images, rows, strict=True)):
        raise ValueError('array identity differs from manifest')
    return images, rows


def job_directory(args):
    return args.out / args.dataset / args.tag


def fit(args):
    images, rows = load_images(args, 'learn', args.limit)
    if args.tag == 'formal' and len(images) != SIZES['learn']:
        raise ValueError('formal fit must use the complete frozen learning split')
    directory = job_directory(args)
    directory.mkdir(exist_ok=True)
    path = directory / (args.transform + '-dictionary.json')
    receipt_path = directory / (args.transform + '-fit.json')
    if path.exists() or receipt_path.exists():
        raise FileExistsError('fit already exists; inspect its receipt before continuing')
    transformed = images
    if args.transform == 'difference':
        transformed = np.stack([codec.difference(image) for image in images])
    samples = list(codec.channel_samples(transformed))
    missing = sorted(set(range(256)) - {value for _, cells in samples for value in cells.values()})
    samples.extend(((1, 1), {(0, 0): value}) for value in missing)
    model = LosslessNDCandidate(2)
    log('fit-start', dataset=args.dataset, transform=args.transform, images=len(images), channels=len(images)*3)
    model.learn(samples, budget=256, score='frequency')
    codec.validate_model(model)
    if len(model.rules) != 256:
        raise ValueError('formal rule budget not reached')
    save_json(path, model.dictionary())
    save_json(receipt_path, {'images': len(images), 'samples': rows, 'rules': len(model.rules),
                             'dictionary_sha256': digest_file(path), 'resource': resource_receipt()})
    log('fit-complete', dataset=args.dataset, transform=args.transform, **resource_receipt())


def torch_cpu():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return torch


def model_for(args, transform):
    directory = job_directory(args)
    path = directory / (transform + '-dictionary.json')
    receipt = read_json(directory / (transform + '-fit.json'))
    if digest_file(path) != receipt['dictionary_sha256']:
        raise ValueError('fitted dictionary changed')
    return LosslessNDCandidate.from_dictionary(read_json(path))


def verify_seal(args):
    seal = read_json(args.out / 'sealed-before-test.json')
    for name, digest in seal['artifacts'].items():
        if digest_file(args.out / name) != digest:
            raise ValueError('artifact changed after test seal: ' + name)


def evaluate(args):
    if args.split == 'test':
        if args.tag != 'formal' or args.limit is not None:
            raise ValueError('formal test must use full split and sealed models')
        verify_seal(args)
    images, identities = load_images(args, args.split, args.limit)
    directory = job_directory(args) / args.split
    directory.mkdir(exist_ok=False)
    models = {name: model_for(args, name) for name in ('raw', 'difference')}
    torch = torch_cpu()
    streams = {method: (directory / (method + '.bf16')).open('xb') for method in METHODS}
    records = []
    try:
        for index, (image, identity) in enumerate(zip(images, identities, strict=True)):
            record = {'identity': identity, 'methods': {}}
            for method in METHODS:
                residual, bpe = method.startswith('difference'), method.endswith('-bpe')
                transformed = codec.difference(image) if residual else image
                model = models['difference' if residual else 'raw']
                blob, counts = codec.encode_bpe(transformed, model) if bpe else (transformed.tobytes(), [1024]*3)
                capacity_positions = {}
                for capacity in CAPACITIES:
                    tensor = codec.to_bf16(blob, capacity, torch)
                    restored = codec.restore_input(tensor, capacity, image.shape, model,
                                                   residual=residual, bpe=bpe, torch=torch)
                    if not np.array_equal(restored, image):
                        raise ValueError('RGB input reconstruction differs')
                    capacity_positions[str(capacity)] = len(tensor)
                    if capacity == 192:
                        streams[method].write(struct.pack('<I', len(tensor)))
                        streams[method].write(tensor.view(torch.uint16).numpy().tobytes())
                record['methods'][method] = {'tokens': sum(counts), 'channel_tokens': counts,
                                             'payload_bytes': len(blob), 'positions': capacity_positions}
            records.append(record)
            if (index + 1) % 100 == 0:
                log('evaluated', dataset=args.dataset, split=args.split, images=index+1)
    finally:
        for stream in streams.values():
            stream.close()
    save_json(directory / 'report.json', {'complete': True, 'images': len(records), 'records': records,
                                          'lossless_per_capacity': len(records)*len(METHODS),
                                          'tensor_files': {method: digest_file(directory / (method+'.bf16'))
                                                           for method in METHODS}, 'resource': resource_receipt()})
    log('evaluate-complete', dataset=args.dataset, split=args.split, images=len(records), **resource_receipt())


def seal(args):
    artifacts = {name: digest_file(args.out / name) for name in ('protocol.json', 'inputs.json')}
    for dataset in SOURCES:
        for transform in ('raw', 'difference'):
            for suffix in ('-dictionary.json', '-fit.json'):
                name = f'{dataset}/formal/{transform}{suffix}'
                artifacts[name] = digest_file(args.out / name)
            receipt = read_json(args.out / f'{dataset}/formal/{transform}-fit.json')
            if receipt['images'] != 1000 or receipt['rules'] != 256:
                raise ValueError('full learning protocol not completed')
        name = f'{dataset}/formal/validation/report.json'
        report = read_json(args.out / name)
        if not report['complete'] or report['images'] != 200:
            raise ValueError('full validation not completed')
        artifacts[name] = digest_file(args.out / name)
    save_json(args.out / 'sealed-before-test.json', {'artifacts': artifacts, 'time_unix': time.time()})
    log('sealed-before-test')


def decode(args):
    if args.split == 'test':
        verify_seal(args)
    directory = job_directory(args) / args.split
    report = read_json(directory / 'report.json')
    samples = read_json(args.out / 'inputs.json')['datasets'][args.dataset]['splits'][args.split]['samples']
    selected = balanced_subset(samples, len(report['records']) if args.tag == 'smoke' else None)
    if [row['identity'] for row in report['records']] != [samples[i] for i in selected]:
        raise ValueError('decoder identities differ from frozen input manifest')
    torch = torch_cpu()
    checked = {}
    for method in METHODS:
        residual, bpe = method.startswith('difference'), method.endswith('-bpe')
        model = model_for(args, 'difference' if residual else 'raw')
        path = directory / (method + '.bf16')
        if digest_file(path) != report['tensor_files'][method]:
            raise ValueError('saved BF16 input changed')
        with path.open('rb') as handle:
            for row in report['records']:
                prefix = handle.read(4)
                if len(prefix) != 4:
                    raise ValueError('truncated tensor record')
                length, = struct.unpack('<I', prefix)
                if not 1 <= length <= 19:
                    raise ValueError('tensor frame outside RGB input bound')
                blob = handle.read(length * 384)
                if len(blob) != length * 384:
                    raise ValueError('truncated BF16 tensor')
                tensor = torch.frombuffer(bytearray(blob), dtype=torch.bfloat16).reshape(length, 192)
                image = codec.restore_input(tensor, 192, (3, 32, 32), model,
                                             residual=residual, bpe=bpe, torch=torch)
                if sha256(image.tobytes()).hexdigest() != row['identity']['sha256']:
                    raise ValueError('independent reconstruction differs from RGB manifest')
                if length != row['methods'][method]['positions']['192']:
                    raise ValueError('saved position count differs')
            if handle.read(1):
                raise ValueError('trailing tensor records')
        checked[method] = len(report['records'])
    save_json(directory / 'independent-decode.json', {'lossless': checked, 'resource': resource_receipt()})
    log('independent-decode-complete', dataset=args.dataset, split=args.split, lossless=checked)


def summarize(args):
    verify_seal(args)
    results = {}
    for dataset in SOURCES:
        directory = args.out / dataset / 'formal' / 'test'
        report = read_json(directory / 'report.json')
        decoded = read_json(directory / 'independent-decode.json')
        if decoded['lossless'] != {method: 1000 for method in METHODS} or len(report['records']) != 1000:
            raise ValueError('independent test decoding incomplete')
        result = {}
        for capacity in CAPACITIES:
            counts = {method: np.array([row['methods'][method]['positions'][str(capacity)]
                                       for row in report['records']], dtype=np.int64) for method in METHODS}
            base = counts['raw']
            methods = {}
            for method, values in counts.items():
                gain = base - values
                rng = np.random.Generator(np.random.PCG64(111404))
                samples = np.concatenate([gain[rng.integers(0, len(gain), (100, len(gain)))].mean(axis=1)
                                          for _ in range(100)])
                batches = [len(values[i:i+16])*int(values[i:i+16].max()) for i in range(0, len(values), 16)]
                methods[method] = {
                    'total_positions': int(values.sum()), 'mean_positions': float(values.mean()),
                    'median_positions': float(np.median(values)), 'p95_positions': float(np.quantile(values, .95)),
                    'mean_activation_bytes': float(values.mean()*384),
                    'fewer_positions_percent': float(100*gain.sum()/base.sum()),
                    'gain_mean_ci95': np.quantile(samples, [.025, .975]).tolist(),
                    'win_tie_loss': [int(np.count_nonzero(gain > 0)), int(np.count_nonzero(gain == 0)),
                                     int(np.count_nonzero(gain < 0))],
                    'batch16_allocated_positions': int(sum(batches)),
                    'at_least_50_60_70_percent_fewer_vs_pixels': [
                        int(np.count_nonzero(values <= 3072*(1-fraction))) for fraction in (.5, .6, .7)],
                }
            result[str(capacity)] = methods
        results[dataset] = result
    save_json(args.out / 'summary.json', {'complete': True, 'results': results, 'resource': resource_receipt()})
    log('summary-complete')


STARTED = time.perf_counter()


def resource_receipt():
    process = psutil.Process()
    memory = process.memory_info()
    return {'seconds': time.perf_counter()-STARTED, 'rss_bytes': memory.rss,
            'peak_rss_bytes': getattr(memory, 'peak_wset', memory.rss),
            'priority': int(process.nice()), 'affinity': process.cpu_affinity(),
            'workers': 0, 'threads_per_library': 1, 'python': sys.version.split()[0], 'numpy': np.__version__}


def memory_watch(args, stop):
    while not stop.wait(.5):
        memory = psutil.Process().memory_info().rss
        if memory > args.max_rss_gib * 1024**3 or psutil.virtual_memory().available < 2.5 * 1024**3:
            path = args.out / f'memory-stop-{args.stage}-{os.getpid()}.json'
            if args.out.exists():
                save_json(path, {'status': 'resource-stop', 'stage': args.stage, 'dataset': args.dataset,
                                 'transform': args.transform, 'resource': resource_receipt()})
            log('resource-stop', **resource_receipt())
            os._exit(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'fit', 'evaluate', 'seal', 'decode', 'summarize'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backup', type=Path)
    parser.add_argument('--old', type=Path)
    parser.add_argument('--dataset', choices=tuple(SOURCES), default='cifar')
    parser.add_argument('--transform', choices=('raw', 'difference'), default='raw')
    parser.add_argument('--tag', choices=('smoke', 'formal'), default='formal')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-rss-gib', type=float, default=8)
    args = parser.parse_args()
    args.out = args.out.resolve()
    process = psutil.Process()
    process.nice(psutil.IDLE_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:1])
    stop = threading.Event()
    watcher = threading.Thread(target=memory_watch, args=(args, stop), daemon=True)
    watcher.start()
    try:
        globals()[args.stage](args)
    finally:
        stop.set()
        watcher.join(timeout=1)


if __name__ == '__main__':
    main()
