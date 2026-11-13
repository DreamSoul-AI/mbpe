"""Sharded execution of the unchanged official replacement/export loop.

Only image selection, process resources and output locations are wrapped.
Functions are extracted verbatim from the existing data wrapper to avoid its
eight-thread import side effects. Neither wrapper nor notebook is rewritten.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
NOTEBOOK = REPO / 'third_party/MDBPE_TF/MDBPE/processor_python.ipynb'
HELPER = SCRIPT.with_name('elsner_svhn_celeba.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def resource_setup(cpu):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMBA_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        os.environ[key] = '1'
    os.environ.update(CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg',
                      PYTHONDONTWRITEBYTECODE='1')
    import psutil
    proc = psutil.Process()
    proc.nice(psutil.IDLE_PRIORITY_CLASS)
    proc.cpu_affinity([cpu])


def worker(args):
    resource_setup(args.cpu)
    os.environ['MPLCONFIGDIR'] = str(args.out / 'mpl')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from elsner_rule_decode import rule_shapes, decode_sequence

    args.out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.inputs / 'input.json').read_text())['channels'][args.channel]
    image_file = Path(meta['files']['test1000']['path'])
    assert digest(image_file) == meta['files']['test1000']['sha256']
    all_images = np.load(image_file)
    positions = json.loads(args.positions)
    images = all_images[positions]
    rules_file = Path(meta['elsner_rules'])
    assert digest(rules_file) == meta['elsner_rules_sha256']
    before = digest(NOTEBOOK)
    # Compile the original wrapper function nodes without its module-level
    # resource setters, downloads, CLI or unrelated dataset entry points.
    names = {'_cells', '_dedent', '_replace_block', '_prepare_setup', '_namespace',
             '_subset_code', '_transcribe', '_apply_test', '_lossless'}
    tree = ast.parse(HELPER.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in names]
    assert {node.name for node in selected} == names
    namespace = dict(Path=Path, os=os, json=json, torch=torch, time=time,
                     shutil=shutil, REPO=REPO, NOTEBOOK=NOTEBOOK, DATA=REPO/'data',
                     PREPARED_GRIDS=[image.astype(np.uint8) for image in images])
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(HELPER), 'exec'), namespace)
    summary = namespace['_apply_test'](args.out/'test', rules_file.parent,
                                        'grids', 1, len(images), 32, 32, 256)
    shapes = rule_shapes(torch.load(rules_file, weights_only=False))
    native_bad, external_bad, lengths = [], [], []
    root = args.out/'test'
    # _apply_test has run the author's original decoder already. Also preserve
    # image-level failures and verify the transcribed integer inputs exactly.
    cells = namespace['_cells']()
    dec_ns = {'torch': torch, 'WIDTH': 32, 'HEIGHT': 32,
              'START_OF_NEW_TOKENS': 256}
    prefix = cells[16].split('def generate_sequence', 1)[0]
    exec(compile(prefix, str(NOTEBOOK)+'::resolve_only', 'exec'), dec_ns)
    for index, image in enumerate(images):
        seq_file = next((root/'tokensequences').glob(f'*/{index}.dat'))
        sequence = torch.load(seq_file, weights_only=False)
        original_file = next((root/'transcribed_data_og').glob(f'*/{index}.dat'))
        original = torch.load(original_file, weights_only=False).long()
        assert torch.equal(original, torch.tensor(image, dtype=torch.long))
        lengths.append(len(sequence))
        if decode_sequence(sequence, shapes, (32, 32)) != image.tolist():
            external_bad.append(positions[index])
        grid = torch.full((32, 32), -1, dtype=torch.long)
        try:
            for token, x, y in sequence:
                grid = dec_ns['resolve_token'](grid, int(token), int(x), int(y))
            if not torch.equal(grid, original):
                native_bad.append(positions[index])
        except Exception:
            native_bad.append(positions[index])
    assert len(native_bad) == summary['lossless_bad']
    assert before == digest(NOTEBOOK)
    report = dict(channel=args.channel, positions=positions, lengths=lengths,
                  native_bad=native_bad, external_bad=external_bad, summary=summary,
                  input_pixel_sha256=hashlib.sha256(images.tobytes()).hexdigest(),
                  notebook_sha256=before, wrapper_sha256=digest(HELPER),
                  rules_sha256=digest(rules_file), cpu_affinity=[args.cpu],
                  torch_threads=torch.get_num_threads(), data_workers=0)
    save(args.out/'report.json', report)
    print('SHARD_DONE', args.channel, positions[0], len(positions), flush=True)


def controller(args):
    import psutil
    allowed = psutil.Process().cpu_affinity()
    assert len(allowed) >= 2
    resource_setup(allowed[0])
    args.out.mkdir(parents=True, exist_ok=True)
    paired = json.loads((args.paired/'report.json').read_text())
    assert len(paired) == 3, 'Complete paired baseline first'
    paired_positions = json.loads((args.inputs/'input.json').read_text())['paired_positions_in_1000']
    if args.validate:
        jobs = [('red', paired_positions[:25], args.out/'validation-red')]
    else:
        validation = json.loads((args.out/'validation-red/report.json').read_text())
        baseline = next(row for row in paired if row['channel'] == 'red')
        expected = [baseline['lengths'][paired_positions.index(p)] for p in validation['positions']]
        assert validation['lengths'] == expected and not validation['native_bad']
        assert not validation['external_bad']
        remaining = sorted(set(range(1000))-set(paired_positions))
        assert len(remaining) == 900
        jobs = [(channel, remaining[i:i+25], args.out/channel/f'shard-{i//25:02d}')
                for channel in ('red', 'green', 'blue') for i in range(0, 900, 25)]
    active, completed = [], []
    free_cpus = allowed[1:1+min(args.workers, 7)]
    while jobs or active:
        while jobs and free_cpus:
            channel, positions, out = jobs.pop(0)
            if (out/'report.json').exists():
                completed.append(out/'report.json')
                continue
            if out.exists():
                raise FileExistsError(f'Incomplete shard preserved: {out}')
            cpu = free_cpus.pop(0)
            out.parent.mkdir(parents=True, exist_ok=True)
            log = out.with_suffix('.log').open('w', encoding='utf-8')
            cmd = [sys.executable, '-B', '-X', 'utf8', '-u', str(SCRIPT),
                   '--worker', '--cpu', str(cpu), '--inputs', str(args.inputs),
                   '--out', str(out), '--channel', channel, '--positions', json.dumps(positions)]
            process = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            active.append((process, log, cpu, out))
            print('START', channel, positions[0], len(positions), 'cpu', cpu, flush=True)
        for entry in active[:]:
            process, log, cpu, out = entry
            result = process.poll()
            if result is None:
                continue
            log.close()
            active.remove(entry)
            free_cpus.append(cpu)
            if result:
                # Let already-started children finish; no unrelated process is touched.
                for running, handle, _, _ in active:
                    running.wait()
                    handle.close()
                raise RuntimeError(f'Shard failed: {out}, code {result}; see log')
            completed.append(out/'report.json')
            print('DONE', len(completed), 'active', len(active), flush=True)
        if active:
            time.sleep(1)
    if args.validate:
        print('VALIDATION_READY', flush=True)
        return
    rows = []
    for baseline in paired:
        channel = baseline['channel']
        lengths = dict(zip(paired_positions, baseline['lengths']))
        assert baseline['summary']['lossless_bad'] == 0
        assert not baseline['external_rule_decode_bad']
        native_bad, external_bad = [], []
        for path in sorted((args.out/channel).glob('shard-*/report.json')):
            shard = json.loads(path.read_text())
            assert shard['notebook_sha256'] == baseline['notebook_sha256']
            assert shard['rules_sha256'] == baseline['rules_sha256']
            for position, length in zip(shard['positions'], shard['lengths'], strict=True):
                assert position not in lengths
                lengths[position] = length
            native_bad.extend(shard['native_bad'])
            external_bad.extend(shard['external_bad'])
        assert set(lengths) == set(range(1000))
        rows.append(dict(channel=channel, images=1000,
                         lengths=[lengths[i] for i in range(1000)],
                         remaining_fraction=sum(lengths.values())/(1000*1024),
                         native_bad=native_bad, external_bad=external_bad,
                         notebook_sha256=baseline['notebook_sha256'],
                         rules_sha256=baseline['rules_sha256']))
    save(args.out/'report.json', rows)
    print('ALL_DONE', [(r['channel'], r['remaining_fraction'], len(r['native_bad'])) for r in rows], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--paired', type=Path)
    parser.add_argument('--workers', type=int, default=7, choices=range(1, 8))
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--channel')
    parser.add_argument('--positions')
    args = parser.parse_args()
    worker(args) if args.worker else controller(args)
