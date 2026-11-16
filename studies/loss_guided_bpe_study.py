"""Compare frequency and measured-loss BPE over multiple CPU-only merge rounds."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import chain
import json
from pathlib import Path
import time

from loss_guided_pair_study import load_images, streams, verify_image

import numpy as np
import psutil


BUDGETS = (0, 1, 8, 16, 32, 64, 128, 256)


class FrozenBytePredictor:
    """A shared count table queried by the last input token's exact byte expansion."""

    def __init__(self, rows):
        joined = np.concatenate([np.r_[np.frombuffer(channel, dtype=np.uint8).astype(np.uint16), 256]
                                 for row in rows for channel in row])
        self.data = joined
        self.locations = {}
        self.distributions = {b"": np.full(256, 1 / 256)}

    def occurrences(self, word):
        if word not in self.locations:
            if len(word) == 1:
                locations = np.flatnonzero(self.data == word[0])
            else:
                locations = self.occurrences(word[:-1])
                locations = locations[locations + len(word) - 1 < len(self.data)]
                locations = locations[self.data[locations + len(word) - 1] == word[-1]]
            self.locations[word] = locations
        return self.locations[word]

    def probability(self, word):
        if word not in self.distributions:
            targets = self.data[self.occurrences(word) + len(word)]
            counts = np.bincount(targets, minlength=257)[:256]
            prior = 128 if len(word) == 1 else 64
            self.distributions[word] = (
                counts + prior * self.probability(word[1:])
            ) / (counts.sum() + prior)
        return self.distributions[word]


def merge(ids, pair, symbol):
    output = []
    left, right = pair
    index = 0
    while index < len(ids):
        if ids[index] == left and index + 1 < len(ids) and ids[index + 1] == right:
            output.append(symbol)
            index += 2
        else:
            output.append(ids[index])
            index += 1
    return output


def pair_counts(sequences, vocabulary):
    width = vocabulary + 1
    flat = np.fromiter(chain.from_iterable(chain(ids, (vocabulary,)) for ids in sequences),
                       dtype=np.int32)
    counts = np.bincount(flat[:-1] * width + flat[1:], minlength=width**2).reshape(width, width)
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1], True])
    symbols = flat[starts[:-1]]
    overlaps = np.bincount(symbols, weights=(np.diff(starts) - 1) // 2, minlength=width)
    diagonal = np.arange(vocabulary)
    counts[diagonal, diagonal] -= overlaps[:vocabulary].astype(np.int64)
    return np.ascontiguousarray(counts[:vocabulary, :vocabulary])


def suffix_pair(ids):
    if len(ids) < 2:
        return None
    left, right = ids[-2:]
    if left != right:
        return left, right
    run = 2
    while run < len(ids) and ids[-1 - run] == right:
        run += 1
    if run % 2 == 0:
        return left, right
    return None


def prefix_samples(rows, rng, per_channel):
    samples = []
    for image, channels in enumerate(rows):
        for channel in channels:
            targets = rng.choice(np.arange(1, len(channel)), size=per_channel, replace=False)
            samples.extend((image, list(channel[max(0, target - 64):target]), channel[target])
                           for target in targets)
    return samples


def measured_deltas(samples, candidates, words, predictor, context_tokens=1):
    if context_tokens == 2:
        return two_token_deltas(samples, candidates, words, predictor)
    counts = {pair: np.zeros(256, dtype=np.int64) for pair in candidates}
    for _, ids, target in samples:
        pair = suffix_pair(ids)
        if pair in counts:
            counts[pair][target] += 1
    results = {}
    for pair, outcomes in counts.items():
        left, right = (words[index] for index in pair)
        delta = np.log(predictor.probability(right)) - np.log(predictor.probability(left + right))
        results[pair] = (float(outcomes @ delta / len(samples)), int(outcomes.sum()))
    return results


def two_token_deltas(samples, candidates, words, predictor):
    changes = {pair: [0.0, 0] for pair in candidates}
    expanded = {pair: words[pair[0]] + words[pair[1]] for pair in candidates}
    for _, ids, target in samples:
        tail = ids[-3:]
        relevant = changes.keys() & set(zip(tail, tail[1:]))
        if not relevant:
            continue
        before = b"".join(words[token] for token in ids[-2:])
        for pair in sorted(relevant):
            converted = merge(ids, pair, len(words))
            after = b"".join(words[token] if token < len(words) else expanded[pair]
                             for token in converted[-2:])
            if before != after:
                changes[pair][0] += (
                    np.log(predictor.probability(before)[target])
                    - np.log(predictor.probability(after)[target])
                )
                changes[pair][1] += 1
    return {pair: (delta / len(samples), count) for pair, (delta, count) in changes.items()}


def evaluate_checkpoints(words, rules, rows, samples, identities, predictor, budgets, context_tokens):
    encoded = rows
    prefixes = samples
    owners = [image for image, _, _ in prefixes]
    image_counts = np.bincount(owners, minlength=len(rows))
    completed = 0
    for budget in budgets:
        for step in range(completed, budget):
            symbol, pair = 256 + step, rules[step]
            encoded = [[merge(ids, pair, symbol) for ids in channels] for channels in encoded]
            prefixes = [(image, merge(ids, pair, symbol), target) for image, ids, target in prefixes]
        completed = budget
        image_losses = np.zeros(len(rows))
        for image, ids, target in prefixes:
            context = b"".join(words[token] for token in ids[-context_tokens:])
            image_losses[image] -= np.log(predictor.probability(context)[target])
        tokens = []
        for channels, identity in zip(encoded, identities, strict=True):
            verify_image(channels, words, identity)
            tokens.append(sum(map(len, channels)))
        yield budget, {"tokens": tokens, "nats_per_byte": (image_losses / image_counts).tolist(),
                       "exact_images": len(rows)}


def learn(rows, score_rows, score_prefixes, predictor, rounds, strategy, context_tokens):
    words = [bytes([symbol]) for symbol in range(256)]
    training = [list(channel) for row in rows for channel in row]
    scoring = [list(channel) for row in score_rows for channel in row]
    prefixes = score_prefixes
    rules = []
    trace = []
    for step in range(1, rounds + 1):
        size = len(words)
        fit = pair_counts(training, size)
        score = pair_counts(scoring, size)
        ranked = np.argsort(-fit.ravel(), kind="stable")[:16]
        candidates = [divmod(int(index), size) for index in ranked]
        frequency_pair = candidates[0]
        eligible = [pair for pair in candidates
                    if fit[pair] > 0 and fit[pair] >= 0.90 * fit[frequency_pair]
                    and score[pair] >= 0.90 * score[frequency_pair]]
        measured = measured_deltas(prefixes, eligible, words, predictor, context_tokens)
        selected = frequency_pair
        if strategy == "loss":
            observed = [pair for pair in eligible if measured[pair][1] >= 32]
            if observed:
                selected = min(observed, key=lambda pair: (measured[pair][0], -fit[pair], pair))
        word = words[selected[0]] + words[selected[1]]
        trace.append({"step": step, "selected": selected, "word": word.hex(),
                      "alternatives": [{"pair": pair, "fit_matches": int(fit[pair]),
                                        "score_matches": int(score[pair]),
                                        "delta_nats": measured[pair][0],
                                        "scored_prefix_matches": measured[pair][1]}
                                       for pair in eligible]})
        training = [merge(ids, selected, size) for ids in training]
        scoring = [merge(ids, selected, size) for ids in scoring]
        prefixes = [(index, merge(ids, selected, size), target) for index, ids, target in prefixes]
        words.append(word)
        rules.append(selected)
        if step == 1 or step % 8 == 0:
            print(f"{strategy}: rule {step}/{rounds}", flush=True)
    return words, rules, trace


def self_check():
    rows = [[7, 7, 7, 7, 7], [7, 7, 3], [3, 7]]
    counts = pair_counts(rows, 8)
    for left in range(8):
        for right in range(8):
            pair = left, right
            expected = sum(len(ids) - len(merge(ids, pair, 8)) for ids in rows)
            if counts[pair] != expected:
                raise ValueError("Pair frequency differs from exact replacements")
    for ids in rows:
        for end in range(1, len(ids) + 1):
            prefix = ids[:end]
            eligible = [pair for pair in Counter(zip(prefix, prefix[1:]))
                        if merge(prefix, pair, 8)[-1] == 8]
            expected = eligible[0] if eligible else None
            if suffix_pair(prefix) != expected:
                raise ValueError("Suffix scoring differs from causal hard merge")


def read_frozen_inputs(entries):
    datasets, identities = {}, {}
    for name, entry in entries.items():
        path = Path(entry["path"])
        if sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"{name} archive differs")
        images = load_images(path)[entry["indices"]]
        if [sha256(image.tobytes()).hexdigest() for image in images] != entry["image_sha256"]:
            raise ValueError(f"{name} images differ")
        datasets[name] = streams(images)
        identities[name] = entry["image_sha256"]
    return datasets, identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=32, choices=BUDGETS[2:])
    parser.add_argument("--prefixes", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=1, choices=(1, 2))
    parser.add_argument("--fit-only", action="store_true", help="Freeze rules for a separate replay")
    parser.add_argument("--seed", type=int, default=171702)
    args = parser.parse_args()
    proc = psutil.Process()
    proc.nice(psutil.IDLE_PRIORITY_CLASS)
    proc.cpu_affinity(proc.cpu_affinity()[:1])
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise MemoryError("Less than 4 GiB of available system RAM")
    if not 1 <= args.prefixes <= 256:
        parser.error("--prefixes must be in 1..256")
    self_check()
    started = time.monotonic()
    rng = np.random.default_rng(args.seed)
    source = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    datasets, identities = read_frozen_inputs(source["inputs"])
    predictor = FrozenBytePredictor(datasets["fit"])
    score_prefixes = prefix_samples(datasets["score"], rng, args.prefixes)
    report = {"seed": args.seed, "prefixes_per_channel": args.prefixes, "rounds": args.rounds,
              "context_tokens": args.context_tokens,
              "inputs": source["inputs"], "branches": {}}
    for strategy in ("frequency", "loss"):
        words, rules, trace = learn(datasets["fit"], datasets["score"], score_prefixes,
                                    predictor, args.rounds, strategy, args.context_tokens)
        report["branches"][strategy] = {"words": [word.hex() for word in words], "trace": trace}
    report["fit_seconds"] = time.monotonic() - started
    report["fit_peak_working_set_bytes"] = proc.memory_info().peak_wset
    (args.output / "choices.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.fit_only:
        return
    for split in ("test", "replication"):
        samples = prefix_samples(datasets[split], rng, args.prefixes)
        evidence = {}
        for strategy, branch in report["branches"].items():
            words = [bytes.fromhex(word) for word in branch["words"]]
            rules = [entry["selected"] for entry in branch["trace"]]
            evidence[strategy] = {}
            budgets = [budget for budget in BUDGETS
                       if budget <= args.rounds and (budget > 0 or strategy == "frequency")]
            for budget, result in evaluate_checkpoints(words, rules, datasets[split], samples,
                                                       identities[split], predictor, budgets,
                                                       args.context_tokens):
                evidence[strategy][budget] = result
                print(f"{split}: {strategy}, {budget} rules", flush=True)
        (args.output / f"{split}.json").write_text(json.dumps(evidence), encoding="utf-8")
    report["elapsed_seconds"] = time.monotonic() - started
    report["rss_bytes_at_end"] = proc.memory_info().rss
    report["peak_working_set_bytes"] = proc.memory_info().peak_wset
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
