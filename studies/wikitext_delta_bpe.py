"""Measure reversible UTF-8 byte differences and minimum-token parsing on WikiText-2."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import threading
from time import perf_counter
from zipfile import ZipFile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.update(TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='')

import numpy as np
import psutil
from tokenizers import Tokenizer, models, trainers
from wikitext_fixed_file import ARCHIVE_SHA256

BUDGETS = (32768, 100000, 151643)
ALPHABET = ''.join(chr(256 + value) for value in range(256))


def digest(data):
    return sha256(data).hexdigest()


def file_digest(path):
    return digest(path.read_bytes())


def chunks(raw, delta):
    values = np.frombuffer(raw, dtype=np.uint8)
    if delta:
        transformed = values.copy()
        transformed[1:] = values[1:] - values[:-1]
        values = transformed
    cursor = 0
    for line in raw.splitlines(keepends=True):
        end = cursor + len(line)
        yield ''.join(ALPHABET[i] for i in values[cursor:end])
        cursor = end
    if cursor != len(raw):
        raise ValueError('line partition changed byte length')


def restore(ids, words, delta):
    data = bytes(ord(c) - 256 for token in ids for c in words[int(token)])
    if delta:
        return np.cumsum(np.frombuffer(data, dtype=np.uint8), dtype=np.uint8).tobytes()
    return data


def save_json(path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def fit(raw, delta, out):
    model = Tokenizer(models.BPE())
    model.train_from_iterator(chunks(raw, delta), trainer=trainers.BpeTrainer(
        vocab_size=BUDGETS[-1], initial_alphabet=list(ALPHABET), min_frequency=1, show_progress=False))
    trained = json.loads(model.to_str())['model']
    records = {}
    for target in BUDGETS:
        merges = trained['merges'][:target - len(ALPHABET)]
        vocab = {word: trained['vocab'][word] for word in ALPHABET}
        vocab.update({a + b: trained['vocab'][a + b] for a, b in merges})
        candidate = Tokenizer(models.BPE(vocab=vocab, merges=[tuple(pair) for pair in merges]))
        path = out / f'{"delta" if delta else "raw"}-{target}.json'
        candidate.save(str(path))
        records[str(target)] = {'file': path.name, 'sha256': file_digest(path), 'entries': len(vocab)}
    return records


def evaluate(args, train, raw):
    import tokenizers
    started = perf_counter()
    models_info = {kind: fit(train, kind == 'delta', args.out) for kind in ('raw', 'delta')}
    save_json(args.out / 'seal.json', {'models': models_info, 'train_sha256': digest(train), 'test_sha256': digest(raw)})
    print('vocabularies sealed', flush=True)
    results = {}

    def record(name, ids, model_info, actual, elapsed):
        if actual != raw:
            raise ValueError(f'test decode differs: {name}')
        path = args.out / f'{name}.npy'
        with path.open('xb') as handle:
            np.save(handle, np.asarray(ids, dtype=np.uint32), allow_pickle=False)
        results[name] = {**model_info, 'tokens': len(ids), 'seconds': elapsed,
                         'packet': path.name, 'packet_sha256': file_digest(path)}
        print(name, len(ids), flush=True)

    qwen = Tokenizer.from_file(str(args.qwen))
    text = raw.decode('utf-8')
    t = perf_counter()
    ids = qwen.encode(text, add_special_tokens=False).ids
    record('qwen3', ids, {'entries': qwen.get_vocab_size(with_added_tokens=False),
                          'sha256': file_digest(args.qwen)},
           qwen.decode(ids, skip_special_tokens=False).encode('utf-8'), perf_counter() - t)
    for kind, specs in models_info.items():
        delta = kind == 'delta'
        strings = list(chunks(raw, delta))
        for target, info in specs.items():
            path = args.out / info['file']
            if file_digest(path) != info['sha256']:
                raise ValueError('model differs from seal')
            bpe = Tokenizer.from_file(str(path))
            vocab = bpe.get_vocab()
            words = {i: w for w, i in vocab.items()}
            minimum = Tokenizer(models.Unigram([(words[i], -1.0) for i in range(len(words))]))
            if minimum.get_vocab() != vocab:
                raise ValueError('minimum parser changed vocabulary IDs')
            counts = []
            for parser, tokenizer in (('bpe', bpe), ('minimum', minimum)):
                t = perf_counter()
                lines = [tokenizer.encode(s, add_special_tokens=False).ids for s in strings]
                elapsed = perf_counter() - t
                counts.append([len(line) for line in lines])
                ids = [token for line in lines for token in line]
                record(f'{kind}-{target}-{parser}', ids, info, restore(ids, words, delta), elapsed)
            if any(a > b for a, b in zip(counts[1], counts[0], strict=True)):
                raise ValueError('minimum parser used more tokens on a line')
    save_json(args.out / 'report.json', {'bytes': len(raw), 'characters': len(text), 'train_bytes': len(train),
        'archive_sha256': ARCHIVE_SHA256, 'test_sha256': digest(raw), 'train_sha256': digest(train),
        'tokenizers': tokenizers.__version__, 'seconds': perf_counter()-started, 'results': results})


def decode(args, raw):
    report = json.loads((args.out / 'report.json').read_bytes())
    results = {}
    for name, row in report['results'].items():
        packet = args.out / row['packet']
        path = args.qwen if name == 'qwen3' else args.out / row['file']
        if file_digest(path) != row['sha256'] or file_digest(packet) != row['packet_sha256']:
            raise ValueError('model or payload changed')
        ids = np.load(packet, allow_pickle=False).tolist()
        if name == 'qwen3':
            actual = Tokenizer.from_file(str(path)).decode(ids, skip_special_tokens=False).encode('utf-8')
        else:
            words = {i: w for w, i in json.loads(path.read_bytes())['model']['vocab'].items()}
            actual = restore(ids, words, name.startswith('delta-'))
        if len(ids) != row['tokens'] or len(actual) != len(raw) or digest(actual) != report['test_sha256'] or actual != raw:
            raise ValueError(f'independent decode differs: {name}')
        results[name] = digest(actual)
    save_json(args.out / 'independent-decode.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('run', 'decode'))
    for name in ('archive', 'qwen', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    process = psutil.Process()
    process.nice(psutil.IDLE_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:1])
    if args.stage == 'run':
        args.out.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            if process.memory_info().rss > 8 * 2**30 or psutil.virtual_memory().available < 4 * 2**30:
                print('resource limit reached', flush=True)
                os._exit(75)
            stop.wait(0.5)

    threading.Thread(target=watch, daemon=True).start()
    if file_digest(args.archive) != ARCHIVE_SHA256:
        raise ValueError('unexpected archive identity')
    with ZipFile(args.archive) as archive:
        raw = archive.read('wikitext-2-raw/wiki.test.raw')
        if args.stage == 'run':
            train = archive.read('wikitext-2-raw/wiki.train.raw')
            evaluate(args, train, raw)
        else:
            decode(args, raw)
    stop.set()


if __name__ == '__main__':
    main()
