"""External harness for unchanged public Being-VL BPE, CPU only."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '8'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import tarfile
import pickle
import time
import traceback
import types

if os.name == 'nt':
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    assert psutil.Process().nice() == psutil.IDLE_PRIORITY_CLASS

extra_deps = os.environ.get('MBPE_PROBE_DEPS')
if extra_deps:
    sys.path.insert(0, extra_deps)
import numpy as np
extra_numba = os.environ.get('MBPE_PROBE_NUMBA_SITE')
if extra_numba:
    sys.path.append(extra_numba)
import numba
numba.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'third_party/Being-VL-0.5'


def load_official():
    manifest = json.loads((SOURCE / 'SOURCE.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location('official_vbpe', SOURCE / 'beingvl/model/vbpe.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def grids(train_per_class=1, test_per_class=1):
    # Read existing archive without extracting files or loading a model.
    with tarfile.open(ROOT / 'data/cifar-10-python.tar.gz') as archive:
        arrays = []
        indices = []
        for member, per_class in (('data_batch_1', train_per_class), ('test_batch', test_per_class)):
            data = pickle.load(archive.extractfile('cifar-10-batches-py/' + member), encoding='bytes')
            labels = np.asarray(data[b'labels'])
            chosen = [int(index) for i in range(10) for index in np.flatnonzero(labels == i)[:per_class]]
            rgb = data[b'data'][chosen].reshape(len(chosen), 3, 32, 32)
            arrays.append(np.ascontiguousarray(rgb[:, 0], dtype=np.int32))
            indices.append(chosen)
    return arrays, indices


def expand(sequence, rules):
    vocab = {int(new): tuple(map(int, pair)) for pair, new in rules}
    result = []
    stack = list(reversed(list(map(int, sequence))))
    while stack:
        token = stack.pop()
        if token in vocab:
            stack.extend(reversed(vocab[token]))
        else:
            result.append(token)
    return np.asarray(result, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['cli', 'api'], required=True)
    parser.add_argument('--train-per-class', type=int, default=1)
    parser.add_argument('--test-per-class', type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    module = load_official()
    (train, test), indices = grids(args.train_per_class, args.test_per_class)
    np.save(args.out / 'train.npy', train)
    np.save(args.out / 'test.npy', test)
    report = {'mode': args.mode, 'source_commit': json.loads((SOURCE/'SOURCE.json').read_text())['commit'],
              'threads': numba.get_num_threads(), 'workers': 0, 'priority': 'IDLE',
              'dataset': 'CIFAR10 red raw pixels', 'indices': indices, 'python': sys.version,
              'numpy': np.__version__, 'numba': numba.__version__, 'train_images': len(train), 'test_images': len(test)}
    start = time.perf_counter()
    try:
        if args.mode == 'cli':
            shim = types.ModuleType('beingvl')
            shim.vBPE = module.vBPE
            sys.modules['beingvl'] = shim
            sys.argv = [str(SOURCE/'beingvl/train/train_vbpe.py'), '--data_path', str(args.out/'train.npy'),
                        '--output_path', str(args.out/'vocab.pkl'), '--num_merges', '256',
                        '--init_size', '256', '--vocab_pad', '0', '--vocab_end', '256', '--validate_data']
            runpy.run_path(sys.argv[0], run_name='__main__')
        else:
            bpe = module.vBPE(256, 0, 256)
            working = train.copy()
            bpe.train(working, 256)
            report['training_input_mutated'] = not np.array_equal(working, train)
            report['rules'] = len(bpe.vbpe_vocab)
            bpe.save_vocab(str(args.out/'vocab.pkl'))
            encoded, decoded, lengths, bad, mutated = [], [], [], [], []
            for i, original in enumerate(test):
                image = original.copy()
                sequence = bpe.merge_token(image)
                rebuilt = expand(sequence, bpe.vbpe_vocab)
                encoded.append(sequence.tolist()); decoded.append(rebuilt.tolist()); lengths.append(len(sequence))
                if not np.array_equal(rebuilt, original.ravel()): bad.append(i)
                if not np.array_equal(image, original): mutated.append(i)
            report.update(lengths=lengths, ratio=sum(lengths)/test.size,
                          reverse_pair_expansion_bad=bad, encoding_mutated_indices=mutated,
                          reverse_pair_expansion_is_not_official_decoder=True)
            # Minimal falsifier: the same horizontal-or-vertical merge loses direction.
            h = np.array([[1, 2], [3, 4]], dtype=np.int32)
            v = np.array([[1, 3], [2, 4]], dtype=np.int32)
            hs = module._apply_merge_2d(h.copy(), 1, 2, 256)
            vs = module._apply_merge_2d(v.copy(), 1, 2, 256)
            report['collision'] = {'input_a': h.tolist(), 'input_b': v.tolist(),
                'output_a': hs.tolist(), 'output_b': vs.tolist(), 'same_output': bool(np.array_equal(hs, vs))}
            one = module.vBPE(256, 0, 256)
            one.train(np.stack([h, v]).copy(), 1)
            ha = one.merge_token(h.copy())
            va = one.merge_token(v.copy())
            report['learned_collision'] = {'rules': one.vbpe_vocab, 'output_a': ha.tolist(),
                'output_b': va.tolist(), 'same_output': bool(np.array_equal(ha, va))}
            for label, sequence in [('equal', [1, 1, 1, 1]), ('descending', [2, 1, 2, 1])]:
                control = module.vBPE(256, 0, 256)
                control.train([np.asarray(sequence, dtype=np.int32)], 1)
                report[label+'_control'] = {'input': sequence, 'rules': control.vbpe_vocab}
            (args.out/'encoded.json').write_text(json.dumps(encoded))
            (args.out/'expanded.json').write_text(json.dumps(decoded))
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    report['seconds'] = time.perf_counter()-start
    load_official()  # source hash verification after execution
    (args.out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return int(report['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
