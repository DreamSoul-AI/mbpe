"""Batch 27: candidate vs Elsner, sample-size curve, scores, bit cost.

Does not change the default of LosslessNDCandidate. Official notebooks and
repositories are only read. New outputs go to the round3 directory.
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
torch.set_num_interop_threads(1)

from lossless_nd_candidate import LosslessNDCandidate
from nd_bpe_incremental_probe import Index
from nd_bpe_probe import add


ELSNER_100 = {
    'red': 96.7060546875,
    'green': 96.515625,
    'blue': 96.6162109375,
}
TIE_PP = 0.1


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def cells_of(image):
    return {(y, x): int(image[y, x]) for y in range(image.shape[0]) for x in range(image.shape[1])}


def h0(values):
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def id_bits(size):
    return max(1, math.ceil(math.log2(max(size, 2))))


def realizable_on(key, index, states, images):
    hits = [(image, a, b) for image, a, b in index.occurrences[key] if image in images]
    if key[0] != key[1]:
        return len(hits)
    occupied = set()
    count = 0
    ordered = []
    for image, a, b in hits:
        ordered.append((states[image].edges[a][b], image, a, b))
    for _, image, a, b in sorted(ordered):
        if (image, a) not in occupied and (image, b) not in occupied:
            occupied.update(((image, a), (image, b)))
            count += 1
    return count


def learn_scored(rank, samples, budget, score):
    """Same replacement contract as LosslessNDCandidate.learn.

    frequency and realizable match that method, including the heap tie break.
    support and holdout only change which key is chosen.
    """
    model = LosslessNDCandidate(rank)
    for shape, cells in samples:
        model._state  # keep attribute use local
        from lossless_nd_candidate import validate_domain
        validate_domain(rank, shape, cells)
    model.base = sorted({v for _, sample_cells in samples for v in sample_cells.values()})
    ids = {v: i for i, v in enumerate(model.base)}
    index = Index()
    states = []
    for image, (shape, sample_cells) in enumerate(samples):
        states.append(model._state(shape, {p: ids[v] for p, v in sample_cells.items()},
                                   index, image, len(model.base)))
    zero = (0,) * rank
    model.shapes = {i: {zero: v} for i, v in enumerate(model.base)}
    model.rules = []
    model.trace = []
    nomination = set(range(len(samples) // 2))
    validation = set(range(len(samples) // 2, len(samples)))
    ties = 0
    for step in range(budget):
        if not index.counts:
            break
        if score == 'frequency':
            key, gain = index.best()
            winners = [item for item, count in index.counts.items() if count == gain]
        elif score == 'realizable':
            gains = {item: model._realizable(item, index, states) for item in index.counts}
            key = min(gains, key=lambda item: (-gains[item], item))
            gain = gains[key]
            winners = [item for item, value in gains.items() if value == gain]
        elif score == 'support':
            support = {item: len({image for image, _, _ in index.occurrences[item]})
                       for item in index.counts}
            key = min(support, key=lambda item: (-support[item], -index.counts[item], item))
            gain = support[key]
            winners = [item for item, value in support.items() if value == gain]
            if gain < 1:
                break
        elif score == 'holdout':
            gains = {item: realizable_on(item, index, states, validation) for item in index.counts}
            nomination_gain = {item: realizable_on(item, index, states, nomination) for item in gains}
            key = min(gains, key=lambda item: (-gains[item], -nomination_gain[item], item))
            gain = gains[key]
            winners = [item for item, value in gains.items() if value == gain]
            # Same floor as the frequency stop (a pair seen once is not a rule).
            if gain < 2:
                break
        else:
            raise ValueError(score)
        if score in ('frequency', 'realizable') and gain < 2:
            break
        if len(winners) > 1:
            ties += 1
        a, b, delta = key
        token = len(model.base) + step
        shape_atoms = dict(model.shapes[a])
        for point, value in model.shapes[b].items():
            shifted = add(point, delta)
            if shifted in shape_atoms:
                raise ValueError('overlapping shape rule')
            shape_atoms[shifted] = value
        if min(shape_atoms) != zero:
            raise ValueError('noncanonical anchor')
        model.shapes[token] = shape_atoms
        model.rules.append((key, token))
        candidates = []
        for image, left, right in index.occurrences[key]:
            state = states[image]
            if state.regions[left][1] > state.regions[right][1]:
                left, right = right, left
            candidates.append((state.edges[left][right], image, left, right))
        saved = 0
        for _, image, left, right in sorted(candidates):
            saved += states[image].replace(left, right, token)
        if score == 'realizable' and saved != gain:
            raise ValueError('gain contract violated')
        model.trace.append({'chosen_score': gain, 'actual_saved': saved, 'tied': len(winners) > 1})
    info = {'ties': ties, 'actual_rules': len(model.rules),
            'score_gap': sum(item['chosen_score'] - item['actual_saved'] for item in model.trace)}
    return model, [len(state.regions) for state in states], info


def roundtrip(model, shape, cells):
    packet = model.encode(shape, cells)
    clone = LosslessNDCandidate.from_dictionary(json.loads(json.dumps(model.dictionary())))
    assert clone.decode(json.loads(json.dumps(packet))) == (tuple(shape), cells)
    return packet


def ratio(lengths, images):
    return sum(lengths) / (images * 1024)


def retention(train_lengths, test_lengths):
    saved_train = 1024 - (sum(train_lengths) / len(train_lengths))
    saved_test = 1024 - (sum(test_lengths) / len(test_lengths))
    return None if saved_train == 0 else saved_test / saved_train


def profile_rules(rules, base_size):
    self_pairs = hierarchical = negative = 0
    for rule in rules:
        if len(rule) == 2 and isinstance(rule[0], (list, tuple)) and len(rule[0]) == 3:
            a, b, delta = rule[0]
            dx, dy = delta
        elif len(rule) == 2 and isinstance(rule[0], list):
            a, b = rule[0]
            dx = dy = None
        else:
            a, b, dx, dy = rule
        self_pairs += int(a == b)
        hierarchical += int(a >= base_size or b >= base_size)
        if dx is not None:
            negative += int(dx < 0 or dy < 0)
    return {'rules': len(rules), 'self_pairs': self_pairs, 'hierarchical': hierarchical,
            'negative_component': negative, 'base_size': base_size}


def load_frozen(inputs):
    meta = json.loads((inputs / 'input.json').read_text(encoding='utf-8'))
    channels = {}
    for name, row in meta['channels'].items():
        arrays = {}
        for label in ('train', 'paired100'):
            path = Path(row['files'][label]['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['files'][label]['sha256']
            arrays[label] = np.load(path)
        channels[name] = {'arrays': arrays, 'rules': Path(row['elsner_rules']),
                          'rules_sha256': row['elsner_rules_sha256']}
    return meta, channels


def class_order_indices(labels, per_class):
    chosen = []
    for label in range(10):
        found = np.flatnonzero(labels == label)[:per_class]
        if len(found) < per_class:
            raise RuntimeError(f'class {label} has only {len(found)} images')
        chosen.extend(int(i) for i in found)
    return chosen


def load_batch(repo):
    with tarfile.open(repo / 'data/cifar-10-python.tar.gz') as archive:
        train = pickle.load(archive.extractfile('cifar-10-batches-py/data_batch_1'), encoding='bytes')
    labels = np.asarray(train[b'labels'])
    pixels = np.asarray(train[b'data']).reshape(-1, 3, 32, 32)
    return labels, pixels


def evaluate(model, images):
    clone = LosslessNDCandidate.from_dictionary(json.loads(json.dumps(model.dictionary())))
    lengths = []
    packets = []
    for image in images:
        packet = model.encode((32, 32), cells_of(image))
        assert clone.decode(json.loads(json.dumps(packet))) == ((32, 32), cells_of(image))
        lengths.append(len(packet['tokens']))
        packets.append(packet)
    return lengths, packets


def run_curve(out, frozen, labels, pixels):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    reference = learn_reference(frozen)
    if not reference:
        raise RuntimeError('custom frequency rules diverged from LosslessNDCandidate.learn')
    for per_class, count in ((1, 10), (3, 30), (10, 100), (30, 300)):
        indices = class_order_indices(labels, per_class)
        if count == 10:
            assert sorted(indices) == sorted(frozen['train_indices'])
        for budget in (64, 256):
            for channel, axis in (('red', 0), ('green', 1), ('blue', 2)):
                started = time.perf_counter()
                train = pixels[indices, axis]
                samples = [((32, 32), cells_of(image)) for image in train]
                model, train_lengths, info = learn_scored(2, samples, budget, 'frequency')
                if count == 10 and budget == 4 and channel == 'red':
                    pass
                test_lengths, packets = evaluate(model, frozen['channels'][channel]['arrays']['paired100'])
                row = {'channel': channel, 'train_images': count, 'per_class': per_class,
                       'budget': budget, 'actual_rules': info['actual_rules'], 'ties': info['ties'],
                       'frequency_overcount': info['score_gap'],
                       'train_ratio': ratio(train_lengths, count),
                       'test_ratio': ratio(test_lengths, 100),
                       'retention': retention(train_lengths, test_lengths),
                       'lossless_bad': 0, 'seconds': time.perf_counter() - started,
                       'train_indices': indices}
                if count == 10 and budget == 256:
                    row['rules'] = model.rules
                    row['base'] = model.base
                    (out / f'{channel}-256-packets.json').write_text(
                        json.dumps([{'tokens': packet['tokens'], 'literals': packet['literals'],
                                     'shape': packet['shape'], 'domain': packet['domain']}
                                    for packet in packets]), encoding='utf-8')
                    (out / f'{channel}-256-dictionary.json').write_text(
                        json.dumps(model.dictionary()), encoding='utf-8')
                rows.append({k: v for k, v in row.items() if k not in ('rules', 'base')})
                if count == 10 and budget == 256:
                    rows[-1]['rules_profile'] = profile_rules(model.rules, len(model.base))
                save(out / 'report.json', {'reference_budget4_matches_stock': reference, 'rows': rows})
                print('P2', channel, count, budget, rows[-1]['test_ratio'], rows[-1]['retention'], flush=True)
    return rows


def learn_reference(frozen):
    image = frozen['channels']['red']['arrays']['train'][:2]
    samples = [((32, 32), cells_of(item)) for item in image]
    stock = LosslessNDCandidate(2)
    stock.learn(samples, budget=4, score='frequency')
    custom, _, _ = learn_scored(2, samples, 4, 'frequency')
    return custom.rules == stock.rules


def p1_judgment(rows):
    judged = []
    for channel, elsner in ELSNER_100.items():
        row = next(item for item in rows if item['channel'] == channel and item['train_images'] == 10
                   and item['budget'] == 256)
        candidate_pp = row['test_ratio'] * 100
        delta = candidate_pp - elsner
        if abs(delta) <= TIE_PP:
            verdict = '持平'
        elif delta < 0:
            verdict = '更短'
        else:
            verdict = '更差'
        judged.append({'channel': channel, 'candidate_percent': candidate_pp,
                       'elsner_percent': elsner, 'delta_percentage_points': delta,
                       'verdict': verdict, 'actual_rules': row['actual_rules'],
                       'ties': row['ties'], 'frequency_overcount': row['frequency_overcount'],
                       'profile': row.get('rules_profile')})
    return judged


def elsner_profiles(frozen):
    profiles = {}
    for channel, row in frozen['channels'].items():
        assert hashlib.sha256(row['rules'].read_bytes()).hexdigest() == row['rules_sha256']
        rules = torch.load(row['rules'], map_location='cpu', weights_only=False)
        profiles[channel] = profile_rules(rules, 256)
        profiles[channel]['first_rule'] = [int(v) for v in rules[0]]
    return profiles


def choose_train_count(rows):
    means = []
    for count in (10, 30, 100, 300):
        subset = [item['test_ratio'] for item in rows if item['train_images'] == count and item['budget'] == 256]
        means.append((sum(subset) / len(subset), count))
    means.sort()
    return means[0][1], [{'train_images': count, 'mean_test_ratio': value} for value, count in sorted(means, key=lambda item: item[1])]


def run_scores(out, labels, pixels, frozen, per_class, budget):
    out.mkdir(parents=True, exist_ok=True)
    blocks = {
        'prefix': (0, per_class),
        'second': (per_class, per_class * 2),
        'third': (per_class * 2, per_class * 3),
    }
    rows = []
    for block, (start, stop) in blocks.items():
        indices = []
        for label in range(10):
            found = np.flatnonzero(labels == label)
            indices.extend(int(i) for i in found[start:stop])
        assert len(indices) == per_class * 10
        for channel, axis in (('red', 0), ('green', 1), ('blue', 2)):
            samples = [((32, 32), cells_of(image)) for image in pixels[indices, axis]]
            test = frozen['channels'][channel]['arrays']['paired100']
            for score in ('frequency', 'realizable', 'support', 'holdout'):
                started = time.perf_counter()
                model, train_lengths, info = learn_scored(2, samples, budget, score)
                test_lengths, _ = evaluate(model, test)
                row = {'block': block, 'channel': channel, 'score': score,
                       'train_images': len(indices), 'budget': budget,
                       'actual_rules': info['actual_rules'], 'ties': info['ties'],
                       'train_ratio': ratio(train_lengths, len(indices)),
                       'test_ratio': ratio(test_lengths, 100),
                       'retention': retention(train_lengths, test_lengths),
                       'lossless_bad': 0, 'seconds': time.perf_counter() - started}
                rows.append(row)
                save(out / 'report.json', rows)
                print('P3', block, channel, score, row['test_ratio'], row['actual_rules'], flush=True)
    return rows


def judge_scores(rows):
    primary = [item for item in rows if item['block'] == 'prefix']
    by_score = {}
    for score in ('frequency', 'realizable', 'support', 'holdout'):
        by_score[score] = {item['channel']: item for item in primary if item['score'] == score}
    verdicts = {}
    for score in ('realizable', 'support', 'holdout'):
        deltas = {}
        not_higher = True
        lower = 0
        same_rules = True
        for channel in ('red', 'green', 'blue'):
            base = by_score['frequency'][channel]
            other = by_score[score][channel]
            delta_pp = (other['test_ratio'] - base['test_ratio']) * 100
            deltas[channel] = delta_pp
            if other['test_ratio'] > base['test_ratio'] + 1e-15:
                not_higher = False
            if delta_pp <= -0.05:
                lower += 1
            if other['actual_rules'] != base['actual_rules']:
                same_rules = False
        positive = not_higher and lower >= 2 and same_rules
        spread = {}
        for channel in ('red', 'green', 'blue'):
            values = [item['test_ratio'] for item in rows if item['score'] == score and item['channel'] == channel]
            spread[channel] = {'min': min(values), 'max': max(values), 'values': values}
        verdicts[score] = {'positive': positive, 'same_rule_count': same_rules,
                           'delta_percentage_points_vs_frequency': deltas, 'spread': spread}
    return verdicts


def bit_table(round2, curve_dir, frozen):
    paired = json.loads((round2 / 'inputs/input.json').read_text(encoding='utf-8'))['paired_positions_in_1000']
    rows = []
    for channel in ('red', 'green', 'blue'):
        raw = frozen['channels'][channel]['arrays']['paired100'].astype(np.uint8)
        raw_bytes = raw.tobytes()
        compressed = zlib.compress(raw_bytes, 9)
        packets = json.loads((curve_dir / f'{channel}-256-packets.json').read_text(encoding='utf-8'))
        dictionary = json.loads((curve_dir / f'{channel}-256-dictionary.json').read_text(encoding='utf-8'))
        tokens = [token for packet in packets for token in packet['tokens']]
        vocab = len(dictionary['base']) + len(dictionary['rules'])
        width = id_bits(vocab)
        dict_bits = 256 + len(dictionary['rules']) * (2 * width + 12)
        shape_bits = 100 * 16
        literal_bits = sum(len(packet['literals']) * (width + 8) for packet in packets)
        domain_bits = 100
        candidate_tokens = h0(tokens) * len(tokens)
        candidate_total = candidate_tokens + dict_bits + shape_bits + literal_bits + domain_bits

        sequences = []
        root = round2 / 'elsner-paired' / channel / 'test' / 'tokensequences'
        for index in range(100):
            sequence = torch.load(root / str(index % 128) / f'{index}.dat', map_location='cpu', weights_only=False)
            sequences.append(sequence)
        elsner_ids = [int(item[0]) for sequence in sequences for item in sequence]
        elsner_rows = [int(item[1]) for sequence in sequences for item in sequence]
        elsner_cols = [int(item[2]) for sequence in sequences for item in sequence]
        rules = torch.load(frozen['channels'][channel]['rules'], map_location='cpu', weights_only=False)
        elsner_vocab = 256 + len(rules)
        elsner_width = id_bits(elsner_vocab)
        elsner_dict = len(rules) * (2 * elsner_width + 12)
        elsner_token_bits = h0(elsner_ids) * len(elsner_ids)
        elsner_pos_bits = h0(elsner_rows) * len(elsner_rows) + h0(elsner_cols) * len(elsner_cols)
        elsner_total = elsner_token_bits + elsner_dict + elsner_pos_bits + shape_bits

        store = np.load(round2 / 'zhang-large' / channel / 'outputs.npz')
        offsets = store['offsets']
        ids = store['ids']
        zhang_ids = []
        for position in paired:
            zhang_ids.extend(int(v) for v in ids[offsets[position]:offsets[position + 1]])
        direction = json.loads((round2 / 'zhang-large' / channel / 'direction_payload.json').read_text(encoding='utf-8'))
        direction_bits = sum(direction['counts'][position] for position in paired)
        zhang_rules = json.loads((round2 / 'zhang-large' / channel / 'rules.json').read_text(encoding='utf-8'))
        zhang_vocab = 1 + max(token for pair, token in zhang_rules)
        zhang_width = id_bits(zhang_vocab)
        zhang_dict = len(zhang_rules) * (2 * zhang_width)
        zhang_token_bits = h0(zhang_ids) * len(zhang_ids)
        zhang_total = zhang_token_bits + zhang_dict + direction_bits + shape_bits
        pixels = 100 * 1024
        rows.append({
            'channel': channel,
            'images': 100,
            'raw_bits': pixels * 8,
            'raw_zero_order_bits': h0(raw_bytes) * len(raw_bytes),
            'zlib9_bytes': len(compressed),
            'zlib9_bits': len(compressed) * 8,
            'candidate': {'tokens': len(tokens), 'token_bits': candidate_tokens,
                          'dictionary_bits': dict_bits, 'shape_bits': shape_bits,
                          'literal_bits': literal_bits, 'domain_flag_bits': domain_bits,
                          'position_bits': 0, 'total_bits': candidate_total,
                          'bits_per_pixel': candidate_total / pixels},
            'elsner': {'tokens': len(elsner_ids), 'token_bits': elsner_token_bits,
                       'dictionary_bits': elsner_dict, 'position_bits': elsner_pos_bits,
                       'shape_bits': shape_bits, 'total_bits': elsner_total,
                       'bits_per_pixel': elsner_total / pixels},
            'zhang': {'tokens': len(zhang_ids), 'token_bits': zhang_token_bits,
                      'dictionary_bits': zhang_dict, 'direction_bits': direction_bits,
                      'shape_bits': shape_bits, 'position_bits': 0,
                      'total_bits': zhang_total, 'bits_per_pixel': zhang_total / pixels},
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--round2', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--skip-scores', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    meta, channels = load_frozen(args.inputs)
    frozen = {'train_indices': meta['train_indices'], 'channels': channels}
    labels, pixels = load_batch(repo)
    curve = run_curve(args.out / 'p1-p2-curve', frozen, labels, pixels)
    judgment = p1_judgment(curve)
    profiles = elsner_profiles(frozen)
    chosen, means = choose_train_count(curve)
    save(args.out / 'p1-p2-curve' / 'judgment.json',
         {'p1': judgment, 'elsner_rule_profile': profiles, 'p2_mean_test': means,
          'chosen_train_images': chosen})
    print('P1', judgment, 'chosen', chosen, flush=True)
    costs = bit_table(args.round2, args.out / 'p1-p2-curve', frozen)
    (args.out / 'p4-bits').mkdir(parents=True, exist_ok=True)
    save(args.out / 'p4-bits' / 'report.json', {'accounting': accounting_note(), 'rows': costs})
    print('P4', [(row['channel'], row['candidate']['bits_per_pixel'], row['elsner']['bits_per_pixel'],
                  row['zhang']['bits_per_pixel'], row['zlib9_bits'] / (100 * 1024)) for row in costs], flush=True)
    if not args.skip_scores:
        per_class = {10: 1, 30: 3, 100: 10, 300: 30}[chosen]
        scored = run_scores(args.out / 'p3-scores', labels, pixels, frozen, per_class, 256)
        save(args.out / 'p3-scores' / 'verdict.json', judge_scores(scored))
        print('P3', judge_scores(scored), flush=True)


def accounting_note():
    return {
        'token_bits': '零阶经验熵乘 token 个数，按该通道 100 张测试的 token 流放在一起计',
        'candidate_dictionary': '256 bit 基字母表出现位图 + 每条规则两个符号号和两个 6 bit 位移',
        'elsner_dictionary': '每条规则两个符号号加两个 6 bit 位移；官方序列另计行、列各自的零阶熵',
        'zhang_dictionary': '公开规则只有两个符号号，没有位移；第一轮方向位按已保存的位数另加',
        'shape': '每张图 32 和 32 各 8 bit',
        'candidate_positions': '位置由锚点顺序推出，记 0',
        'zlib': '同一 100 张 uint8 像素按 zlib 等级 9 压缩后的字节数',
        'amortization': '字典按这 100 张分摊一次，不按每张图重复',
    }


if __name__ == '__main__':
    main()
