"""Official Elsner notebook on CIFAR red masks. The notebook file is not modified.

Settings are substituted only in the in-memory copy of the cells. New token
ids still start at max(mask)+1, which is 2 or 4, so the token ceiling is set
to that start plus the requested rule count.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMBA_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
import hashlib
import json
import pickle
import shutil
import tarfile
import time
from pathlib import Path

import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
import numpy as np
import torch
torch.set_num_threads(1)

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / 'third_party' / 'MDBPE_TF' / 'MDBPE' / 'processor_python.ipynb'


def cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    return [''.join(cell.get('source', [])) for cell in notebook['cells'] if cell.get('cell_type') == 'code']


def dedent(text):
    lines = text.splitlines()
    indents = [len(line) - len(line.lstrip(' ')) for line in lines if line.strip()]
    cut = min(indents) if indents else 0
    return '\n'.join(line[cut:] if len(line) >= cut else line for line in lines)


def replace_block(cell):
    start = cell.index('    it = 0\n    total_saved = 0\n    for unique, classes, path in unique_train:')
    end = cell.index('    #increment token ID')
    return dedent(cell[start:end])


def mask_stack(images, kind):
    if kind == 'bin128':
        return [(image >= 128).astype(np.int64) for image in images]
    if kind == '4level':
        return [(image // 64).astype(np.int64) for image in images]
    raise ValueError(kind)


def class_indices(labels, per_class):
    chosen = []
    for label in range(10):
        found = np.flatnonzero(labels == label)[:per_class]
        chosen.extend(int(index) for index in found)
    return chosen


def load_red(inputs, train_count, test_split, test_count):
    meta = json.loads((inputs / 'input.json').read_text(encoding='utf-8'))
    with tarfile.open(REPO / 'data/cifar-10-python.tar.gz') as archive:
        batch = pickle.load(archive.extractfile('cifar-10-batches-py/data_batch_1'), encoding='bytes')
    labels = np.asarray(batch[b'labels'])
    pixels = np.asarray(batch[b'data']).reshape(-1, 3, 32, 32)
    indices = class_indices(labels, 30)
    assert sorted(class_indices(labels, 1)) == sorted(meta['train_indices'])
    train = list(pixels[indices[:train_count], 0])
    row = meta['channels']['red']
    test = np.load(row['files'][test_split]['path'])[:test_count]
    return train, list(test)


def prepare_setup(source, images, rules, alphabet):
    source = source.replace('WIDTH, HEIGHT = 12, 12', 'WIDTH, HEIGHT = 32, 32', 1)
    ceiling = int(alphabet + rules)
    source = source.replace(
        'MAXIMUM_NUMBER_OF_TOKENS = 256+32 #256 = only use the base tokens, 256+32 --> use 32 extra tokens to compact stuff into',
        f'MAXIMUM_NUMBER_OF_TOKENS = {ceiling} # mask alphabet {alphabet} plus {int(rules)} rules',
        1,
    )
    source = source.replace(
        'LIMITED_EXAMPLES = 100 #--> only use a subset of the data (for debugging); set to "None" to use all data',
        f'LIMITED_EXAMPLES = {int(images)}',
        1,
    )
    return source


def dataset_source():
    return """
class _MaskGrids(torch.utils.data.Dataset):
    def __init__(self, grids):
        self.grids = grids
    def __len__(self):
        return len(self.grids)
    def __getitem__(self, index):
        grid = torch.as_tensor(self.grids[index], dtype=torch.float32) / 255.0
        return grid.unsqueeze(0), 0
mnist_trainset = _MaskGrids(PREPARED_GRIDS)
train_loader = DataLoader(mnist_trainset, batch_size=8, shuffle=False, num_workers=0)
test_loader = train_loader
CHOSEN_INDICES = list(range(len(PREPARED_GRIDS)))
"""


def namespace():
    space = {'__name__': '__main__'}
    exec(compile(cells()[0], f'{NOTEBOOK}::cell0', 'exec'), space, space)
    space['plt'].show = lambda *args, **kwargs: None
    return space


def transcribe(space, source_cells, grids, rules, alphabet):
    space['PREPARED_GRIDS'] = grids
    exec(compile(prepare_setup(source_cells[1], len(grids), rules, alphabet), f'{NOTEBOOK}::cell1', 'exec'), space, space)
    dataset_cell = source_cells[2].replace(
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "DEVICE = torch.device('cpu')",
        1,
    )
    dataset_cell = dataset_cell.replace(
        "mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)\n"
        "mnist_testnset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)\n"
        "train_loader = DataLoader(mnist_trainset, batch_size=8, shuffle=False)\n"
        "test_loader = DataLoader(mnist_testnset, batch_size=8, shuffle=False)\n",
        dataset_source(),
        1,
    )
    show_at = dataset_cell.find('def show(')
    dataset_cell = dataset_cell[:show_at] + 'pass\n'
    exec(compile(dataset_cell, f'{NOTEBOOK}::cell2', 'exec'), space, space)
    for index in (3, 4, 5, 6, 7, 8, 9):
        exec(compile(source_cells[index], f'{NOTEBOOK}::cell{index}', 'exec'), space, space)
    return space


def check_grids(grids):
    modulo = 128
    for index, grid in enumerate(grids):
        saved = torch.load(f'transcribed_data_og/{index % modulo}/{index}.dat', weights_only=False).long().numpy()
        if saved.shape != grid.shape or not np.array_equal(saved, grid):
            raise SystemExit(f'mask grid {index} was not stored exactly')


def lossless(space, count):
    bad = 0
    modulo = int(space['FILE_MODULO'])
    width = int(space['WIDTH'])
    height = int(space['HEIGHT'])
    for index in range(count):
        sequence = torch.load(f'tokensequences/{index % modulo}/{index}.dat', weights_only=False)
        grid = torch.ones(width, height, dtype=torch.long) * -1
        try:
            for token, x_pos, y_pos in sequence:
                grid = space['resolve_token'](grid, int(token), int(x_pos), int(y_pos))
            original = torch.load(f'transcribed_data_og/{index % modulo}/{index}.dat', weights_only=False).long()
            if grid.shape != original.shape or not torch.equal(grid, original):
                bad += 1
        except Exception:
            bad += 1
    return bad


def learn(work, grids, rules, alphabet):
    before = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    os_cwd = Path.cwd()
    os.chdir(work)
    try:
        started = time.perf_counter()
        source_cells = cells()
        space = namespace()
        transcribe(space, source_cells, grids, rules, alphabet)
        check_grids(grids)
        if int(space['NEW_TOKEN_ID']) != alphabet:
            raise SystemExit(f"official start {space['NEW_TOKEN_ID']} != alphabet {alphabet}")
        print('learn images', len(grids), 'codewords', int(space['CODEWORDS_TO_ADD']), flush=True)
        exec(compile(source_cells[10], f'{NOTEBOOK}::cell10', 'exec'), space, space)
        for index in (11, 12, 13, 14, 15):
            exec(compile(source_cells[index], f'{NOTEBOOK}::cell{index}', 'exec'), space, space)
        resolve_source = source_cells[16].replace(
            'show(generate_sequence(WIDTH, HEIGHT)[None].float() / 255.0)', 'pass',
        )
        exec(compile(resolve_source, f'{NOTEBOOK}::cell16', 'exec'), space, space)
        lengths = [int(item) for item in space['sequence_lengths']]
        bad = lossless(space, len(grids))
        payload = {
            'split': 'train', 'images': len(grids), 'rules': len(space['rules']),
            'requested_rules': rules, 'alphabet': alphabet,
            'start_of_new_tokens': int(space['START_OF_NEW_TOKENS']),
            'tokens': int(sum(lengths)), 'ratio': sum(lengths) / (len(grids) * 1024),
            'lossless_bad': bad, 'seconds': time.perf_counter() - started,
            'notebook_sha256': before,
        }
        (work / 'summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print('TRAIN', json.dumps(payload), flush=True)
        return payload
    finally:
        os.chdir(os_cwd)
        after = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
        if after != before:
            raise SystemExit('notebook file changed')


def apply_test(work, rules_dir, grids, rules, alphabet):
    before = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    os_cwd = Path.cwd()
    os.chdir(work)
    try:
        started = time.perf_counter()
        source_cells = cells()
        space = namespace()
        transcribe(space, source_cells, grids, rules, alphabet)
        check_grids(grids)
        anchor_source = source_cells[10]
        anchor_end = anchor_source.index('\nCODEWORDS_ADDED')
        exec(compile(anchor_source[:anchor_end], f'{NOTEBOOK}::get_anchor', 'exec'), space, space)
        learned = torch.load(rules_dir / 'rules.dat', weights_only=False)
        train_meta = json.loads((rules_dir / 'summary.json').read_text(encoding='utf-8'))
        space['START_OF_NEW_TOKENS'] = int(train_meta['start_of_new_tokens'])
        program = compile(replace_block(source_cells[10]), f'{NOTEBOOK}::replace', 'exec')
        missed = 0
        for index, rule in enumerate(learned):
            space['max_key'] = rule
            space['NEW_TOKEN_ID'] = int(train_meta['start_of_new_tokens']) + index
            space['unique_train'] = space['UniqueDataset']()
            space['last_output'] = time.time()
            exec(program, space, space)
            if int(space['total_saved']) <= 0:
                missed += 1
            if index == 0 or (index + 1) % 16 == 0 or index + 1 == len(learned):
                print(f'applied {index + 1}/{len(learned)} missed={missed}', flush=True)
        for index in (11, 12, 13, 14, 15):
            exec(compile(source_cells[index], f'{NOTEBOOK}::cell{index}', 'exec'), space, space)
        resolve_source = source_cells[16].replace(
            'show(generate_sequence(WIDTH, HEIGHT)[None].float() / 255.0)', 'pass',
        )
        exec(compile(resolve_source, f'{NOTEBOOK}::cell16', 'exec'), space, space)
        lengths = [int(item) for item in space['sequence_lengths']]
        bad = lossless(space, len(grids))
        payload = {
            'split': 'test', 'images': len(grids), 'rules': len(learned),
            'rules_missed': missed, 'tokens': int(sum(lengths)),
            'ratio': sum(lengths) / (len(grids) * 1024),
            'lossless_bad': bad, 'seconds': time.perf_counter() - started,
            'notebook_sha256': hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        }
        (work / 'summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print('TEST', json.dumps(payload), flush=True)
        return payload
    finally:
        os.chdir(os_cwd)
        if hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest() != before:
            raise SystemExit('notebook file changed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mask', choices=('bin128', '4level'), required=True)
    parser.add_argument('--train-images', type=int, required=True)
    parser.add_argument('--rules', type=int, required=True)
    parser.add_argument('--test-split', choices=('paired100', 'test1000'), default='paired100')
    parser.add_argument('--test-images', type=int, required=True)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    alphabet = 2 if args.mask == 'bin128' else 4
    train_raw, test_raw = load_red(args.inputs, args.train_images, args.test_split, args.test_images)
    train = mask_stack(train_raw, args.mask)
    test = mask_stack(test_raw, args.mask)
    tag = f'{args.mask}-train{args.train_images}-r{args.rules}-{args.test_split}{args.test_images}'
    root = args.out / 'p18-elsner' / tag
    learn(root / 'train', train, args.rules, alphabet)
    if args.probe:
        apply_test(root / 'probe-test', root / 'train', test, args.rules, alphabet)
        print('PROBE_DONE', flush=True)
        return
    apply_test(root / 'test', root / 'train', test, args.rules, alphabet)
    print('P18_DONE', args.mask, flush=True)


if __name__ == '__main__':
    import os
    main()
