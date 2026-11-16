"""Train the library frequency baseline or replay frozen loss-guided merges."""

import argparse
from hashlib import sha256
from itertools import islice
import json
import os
from pathlib import Path
import time

from loss_guided_bpe_study import BUDGETS, FrozenBytePredictor, prefix_samples, read_frozen_inputs
from loss_guided_pair_study import verify_image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import psutil
import tokenizers
from tokenizers import Tokenizer, models, trainers


def text_bytes(values):
    return "".join(chr(256 + value) for value in values)


def library_baseline(rows, output):
    alphabet = [chr(256 + value) for value in range(256)]
    encoder = Tokenizer(models.BPE())
    encoder.train_from_iterator(
        (text_bytes(channel) for row in rows for channel in row),
        trainer=trainers.BpeTrainer(vocab_size=512, min_frequency=1,
                                    initial_alphabet=alphabet, show_progress=False),
    )
    encoder.save(str(output / "full-tokenizer.json"))
    model = json.loads(encoder.to_str())["model"]
    return (model["vocab"], [tuple(pair) for pair in model["merges"]])


def frozen_vocabulary(branch):
    words = [text_bytes(bytes.fromhex(word)) for word in branch["words"]]
    if len(words) != len(set(words)):
        raise ValueError("Tokenizers replay requires unique token expansions")
    pairs = [(words[a], words[b]) for a, b in (row["selected"] for row in branch["trace"])]
    return ({word: index for index, word in enumerate(words)}, pairs)


def verify_case(vocabulary, rules, budget, text_rows, samples, identities, predictor, context_size):
    pairs = rules[:budget]
    if len(pairs) != budget:
        raise ValueError("Incomplete conventional BPE rule list")
    vocab = {chr(256 + byte): vocabulary[chr(256 + byte)] for byte in range(256)}
    vocab.update({left + right: vocabulary[left + right] for left, right in pairs})
    encoder = Tokenizer(models.BPE(vocab=vocab, merges=pairs))
    words = {token: bytes(ord(c) - 256 for c in word) for word, token in vocab.items()}
    totals = np.zeros(len(text_rows))
    counts = np.bincount([owner for owner, _, _ in samples], minlength=len(text_rows))
    cache = {}
    sequential = iter(samples)
    while batch := list(islice(sequential, 2048)):
        encoded = encoder.encode_batch([text for _, text, _ in batch], add_special_tokens=False)
        for (owner, _, target), packet in zip(batch, encoded, strict=True):
            ids = tuple(packet.ids[-context_size:])
            if ids not in cache:
                context = b"".join(words[token] for token in ids)
                cache[ids] = np.log(predictor.probability(context))
            totals[owner] -= cache[ids][target]
    positions = []
    for channels, identity in zip(text_rows, identities, strict=True):
        ids = [packet.ids for packet in encoder.encode_batch(channels, add_special_tokens=False)]
        verify_image(ids, words, identity)
        positions.append(sum(map(len, ids)))
    return {"tokens": positions, "nats_per_byte": (totals / counts).tolist(),
            "exact_images": len(text_rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--protocol", type=Path, help="Train the standard library baseline")
    source.add_argument("--choices", type=Path, help="Replay frozen custom choices")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    process = psutil.Process()
    process.nice(psutil.IDLE_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:1])
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise MemoryError("Less than 4 GiB of available system RAM")
    source_bytes = (args.protocol or args.choices).read_bytes()
    document = json.loads(source_bytes)
    datasets, identities = read_frozen_inputs(document["inputs"])
    args.output.mkdir(parents=True, exist_ok=False)
    if args.protocol:
        vocabularies = {"standard": library_baseline(datasets["fit"], args.output)}
        budgets, seed, per_channel, context_size = (32, 64, 128, 256), 171717, 128, 1
    else:
        vocabularies = {name: frozen_vocabulary(branch) for name, branch in document["branches"].items()}
        budgets = [budget for budget in BUDGETS if budget <= document["rounds"]]
        seed = document["seed"]
        per_channel = document["prefixes_per_channel"]
        context_size = document.get("context_tokens", 1)
    predictor = FrozenBytePredictor(datasets["fit"])
    rng = np.random.default_rng(seed)
    prefix_samples(datasets["score"], rng, per_channel)
    for split in ("test", "replication"):
        rows = datasets[split]
        samples = [(image, text_bytes(ids), target)
                   for image, ids, target in prefix_samples(rows, rng, per_channel)]
        text_rows = [tuple(text_bytes(channel) for channel in row) for row in rows]
        evidence = {}
        for name, (vocab, rules) in vocabularies.items():
            evidence[name] = {}
            for budget in budgets:
                if name == "loss" and budget == 0:
                    continue
                evidence[name][budget] = verify_case(
                    vocab, rules, budget, text_rows, samples, identities[split], predictor, context_size
                )
                print(f"{split}: {name}, {budget} BPE rules", flush=True)
        if args.protocol:
            evidence = evidence["standard"]
        (args.output / f"{split}.json").write_text(json.dumps(evidence), encoding="utf-8")
    receipt = {**document, "source_sha256": sha256(source_bytes).hexdigest(),
               "tokenizers": tokenizers.__version__, "elapsed_seconds": time.monotonic() - started,
               "peak_working_set_bytes": process.memory_info().peak_wset,
               "rss_bytes_at_end": process.memory_info().rss}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
