"""Minimal Elsner shape-export failure on a hand-built non-convex token.

The notebook cells are executed unchanged from their source strings. The
notebook file is not written.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
import hashlib
import json
from pathlib import Path

import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
import torch
torch.set_num_threads(1)


NOTEBOOK = Path(__file__).resolve().parents[1] / 'third_party/MDBPE_TF/MDBPE/processor_python.ipynb'


def cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    return [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    before = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    # Four 1-pixels form an L whose anchor is not the leftmost column.
    # Rows are the first index, matching the notebook's unique[x, y].
    original = torch.tensor([
        [9, 1, 1, 9],
        [1, 1, 9, 9],
        [9, 9, 9, 9],
        [9, 9, 9, 9],
    ], dtype=torch.long)
    classes = original.clone()
    unique = torch.arange(16, dtype=torch.long).reshape(4, 4)
    token = 256
    member = original == 1
    classes[member] = token
    unique[member] = 100
    code = cells()
    export = next(source for source in code if 'offset_x = most_left - x' in source)
    sequence = next(source for source in code if 'sequence.append([classes[x][y].item(), x, y])' in source)
    resolve = next(source for source in code if 'def resolve_token' in source).split('def generate_sequence', 1)[0]
    (args.out / 'tokensequences' / '0').mkdir(parents=True)
    torch.save(original, args.out / 'transcribed_data_og.dat')
    previous = Path.cwd()
    os.chdir(args.out)
    try:
        namespace = {'torch': torch, 'os': os, 'WIDTH': 4, 'HEIGHT': 4, 'FILE_MODULO': 1,
                     'START_OF_NEW_TOKENS': 256, 'unique_train': [(unique, classes, 'transcribed_data.dat')],
                     'visualise_unique': lambda *unused: None}
        exec(compile(export, str(NOTEBOOK) + '::export', 'exec'), namespace)
        exec(compile(sequence, str(NOTEBOOK) + '::sequence', 'exec'), namespace)
        exec(compile(resolve, str(NOTEBOOK) + '::resolve', 'exec'), namespace)
        saved = torch.load('tokenshapes/256.dat', weights_only=False)
        official = torch.load('tokensequences/0/0.dat', weights_only=False)
        grid = torch.full((4, 4), -1, dtype=torch.long)
        for token_id, x, y in official:
            grid = namespace['resolve_token'](grid, int(token_id), int(x), int(y))
    finally:
        os.chdir(previous)
    anchor = (0, 1)
    relative = {(0, 0): 1, (0, 1): 1, (1, -1): 1, (1, 0): 1}
    external = torch.full((4, 4), -1, dtype=torch.long)
    for token_id, x, y in official:
        if int(token_id) < 256:
            external[int(x), int(y)] = int(token_id)
            continue
        for (dx, dy), value in relative.items():
            external[int(x) + dx, int(y) + dy] = value
    mismatch = (grid != original).nonzero().tolist()
    report = {
        'notebook_sha256': before,
        'notebook_unchanged': before == hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        'original': original.tolist(),
        'classes': classes.tolist(),
        'anchor': list(anchor),
        'column_offset_not_stored': -1,
        'official_offset_x': int(saved[1]),
        'official_crop': saved[0].tolist(),
        'official_sequence': [[int(v) for v in item] for item in official],
        'official_restored': grid.tolist(),
        'official_mismatch_positions': mismatch,
        'external_restored': external.tolist(),
        'external_matches': bool(torch.equal(external, original)),
        'official_matches': bool(torch.equal(grid, original)),
    }
    (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('P5', 'official_match', report['official_matches'], 'mismatch', mismatch,
          'offset_x', report['official_offset_x'], 'external', report['external_matches'], flush=True)
    if report['official_matches'] or not report['external_matches']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
