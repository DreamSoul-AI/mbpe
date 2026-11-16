"""Measure single-rule BPE perturbations with a frozen byte-count predictor."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import psutil

from color_template_codec import color_basis, inverse_color_basis
from rgb_input_codec import difference, restore_difference


def load_images(path):
    with np.load(path, allow_pickle=False) as archive:
        images = archive["images"]
    if images.dtype != np.uint8 or images.shape[1:] != (3, 32, 32):
        raise ValueError(f"Expected uint8 CIFAR-10 images: {path}")
    return images


def streams(images):
    return [tuple(channel.T.tobytes() for channel in difference(color_basis(image)))
            for image in images]


def verify_image(encoded, words, identity):
    channels = [b"".join(words[token] for token in ids) for ids in encoded]
    if any(len(channel) != 1024 for channel in channels):
        raise ValueError("Incorrect decoded channel length")
    residual = np.frombuffer(b"".join(channels), dtype=np.uint8).reshape(3, 32, 32)
    image = inverse_color_basis(restore_difference(residual.swapaxes(-1, -2)))
    if sha256(image.tobytes()).hexdigest() != identity:
        raise ValueError("Independent RGB decode differs")


def fit_counts(rows):
    values = np.asarray([[np.frombuffer(channel, dtype=np.uint8) for channel in row]
                         for row in rows], dtype=np.int32).reshape(-1, 1024)
    pairs = (values[:, :-1] * 256 + values[:, 1:]).ravel()
    order_one = np.bincount(pairs, minlength=256**2).reshape(256, 256)
    triples = (values[:, :-2] * 256**2 + values[:, 1:-1] * 256 + values[:, 2:]).ravel()
    order_two = np.bincount(triples, minlength=256**3).reshape(256**2, 256)
    frequencies = order_one.ravel().copy()
    for symbol in range(256):
        word = bytes([symbol, symbol])
        frequencies[symbol * 257] = sum(channel.count(word) for row in rows for channel in row)
    probabilities = (order_one + 0.5) / (order_one.sum(axis=1, keepdims=True) + 128)
    return probabilities, order_two, frequencies


def evaluate(pair, rows, identities, probabilities, counts, prior=64):
    word = pair.to_bytes(2, "big")
    final_byte = word[-1]
    merged = (counts[pair] + prior * probabilities[final_byte]) / (counts[pair].sum() + prior)
    delta_table = np.log(probabilities[final_byte]) - np.log(merged)
    pattern = re.compile(re.escape(word))
    positions = np.zeros(len(rows), dtype=np.int32)
    deltas = np.zeros(len(rows))
    words = [bytes([symbol]) for symbol in range(256)] + [word]
    for index, row in enumerate(rows):
        encoded = []
        for channel in row:
            ids = []
            cursor = 0
            for match in pattern.finditer(channel):
                start, end = match.span()
                ids.extend(channel[cursor:start])
                ids.append(256)
                cursor = end
                positions[index] += 1
                if end < len(channel):
                    deltas[index] += delta_table[channel[end]]
            ids.extend(channel[cursor:])
            encoded.append(ids)
        if sum(map(len, encoded)) != 3072 - positions[index]:
            raise ValueError("Position total differs from encoded stream")
        verify_image(encoded, words, identities[index])
    return {"positions_saved": positions.tolist(), "loss_delta_nats": deltas.tolist(),
            "exact_images": len(rows)}


def mean_loss(result):
    # The first byte per channel uses an identical start distribution in both arms.
    return float(np.mean(result["loss_delta_nats"]) / 3072)


def summarize(groups, frequencies, evidence):
    score = evidence["score"]
    summaries = []
    for group in groups:
        by_frequency = group[0]
        minimum = 0.85 * np.mean(score[by_frequency]["positions_saved"])
        maximum = 1.15 * np.mean(score[by_frequency]["positions_saved"])
        pool = [pair for pair in group
                if minimum <= np.mean(score[pair]["positions_saved"]) <= maximum]
        by_loss = min(pool, key=lambda pair: (mean_loss(score[pair]), -frequencies[pair], pair))
        summaries.append({"frequency": by_frequency, "loss": by_loss, "matched_pool": pool})
    return summaries


def self_check():
    images = np.zeros((1, 3, 32, 32), dtype=np.uint8)
    rows = streams(images)
    hashes = [sha256(images[0].tobytes()).hexdigest()]
    probabilities = np.full((256, 256), 1 / 256)
    counts = np.zeros((1, 256), dtype=np.int64)
    result = evaluate(0, rows, hashes, probabilities, counts)
    if result["positions_saved"] != [1536] or result["loss_delta_nats"] != [0.0]:
        raise ValueError("Repeated byte pair, expansion or zero-delta check failed")
    # Check that the byte at each scored boundary was outside its tokenized prefix.
    word, channel = b"aa", b"aaabaa"
    ends = [match.end() for match in re.finditer(word, channel)]
    prefix_ends = [n for n in range(1, len(channel) + 1)
                   if list(re.finditer(word, channel[:n]))
                   and list(re.finditer(word, channel[:n]))[-1].end() == n]
    if ends != [2, 6] or prefix_ends != ends:
        raise ValueError("Whole-stream boundaries differ from causal prefix boundaries")


def run(args):
    started = time.monotonic()
    rng = np.random.default_rng(args.seed)
    selections = {"fit": (args.fit, 1000), "score": (args.score, 128),
                  "test": (args.test, 128), "replication": (args.test, 372)}
    datasets, identities, input_info = {}, {}, {}
    used = set()
    for name, (path, size) in selections.items():
        images = load_images(path)
        indices = rng.permutation(len(images)) if name != "replication" else remaining
        chosen = []
        hashes = []
        for index in indices:
            identity = sha256(images[index].tobytes()).hexdigest()
            if identity not in used:
                used.add(identity)
                hashes.append(identity)
                chosen.append(int(index))
            if len(chosen) == size:
                break
        if len(chosen) != size:
            raise ValueError(f"Too few disjoint images in {name}")
        if name == "test":
            remaining = [int(index) for index in indices if index not in chosen]
        datasets[name] = streams(images[chosen])
        identities[name] = hashes
        input_info[name] = {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest(),
                            "indices": chosen, "image_sha256": hashes}
    probabilities, counts, frequencies = fit_counts(datasets.pop("fit"))
    ranked = np.argsort(-frequencies, kind="stable")
    groups = [list(map(int, ranked[start:start + 8])) for start in (0, 32, 128, 512)]
    candidates = [pair for group in groups for pair in group]
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"seed": args.seed, "prior": args.prior, "inputs": input_info, "groups": groups,
               "nonoverlap_fit_count": {pair: int(frequencies[pair]) for pair in candidates}}
    (args.output / "protocol.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    evidence = {}
    for name, rows in datasets.items():
        evidence[name] = {pair: evaluate(pair, rows, identities[name], probabilities, counts, args.prior)
                          for pair in candidates}
        print(f"{name}: {len(rows)} images × {len(candidates)} candidates", flush=True)
        if name == "score":
            choices = summarize(groups, frequencies, evidence)
            (args.output / "choices.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")
        (args.output / f"{name}.json").write_text(json.dumps(evidence[name]), encoding="utf-8")
    receipt["seconds"] = time.monotonic() - started
    receipt["rss_bytes_at_end"] = psutil.Process().memory_info().rss
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--score", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=171701)
    parser.add_argument("--prior", type=float, default=64)
    args = parser.parse_args()
    process = psutil.Process()
    process.nice(psutil.IDLE_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:1])
    self_check()
    if args.self_check:
        print("single-rule roundtrip and causal boundary checks passed")
    else:
        if any(value is None for value in (args.fit, args.score, args.test, args.output)):
            parser.error("--fit, --score, --test and --output are required")
        if not np.isfinite(args.prior) or args.prior <= 0:
            parser.error("--prior must be finite and positive")
        if psutil.virtual_memory().available < 4 * 1024**3:
            raise MemoryError("Less than 4 GiB of available system RAM")
        run(args)


if __name__ == "__main__":
    main()
