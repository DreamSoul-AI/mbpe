"""Compare frozen Qwen3 BPE and WikiText-trained BPE on the same test text."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZipFile

import psutil

from bpe_block_file import BLOCK
from canonical_huffman import lengths_from_counts
from wikitext_fixed_file import ARCHIVE_SHA256, read_tokens, write_vocabulary
from wikitext_huffman_file import TEST_MAGIC, read_bundle, write_bundle


def train_tokenizer(qwen, training_text, vocab_size):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = qwen.normalizer
    tokenizer.pre_tokenizer = qwen.pre_tokenizer
    tokenizer.decoder = qwen.decoder
    tokenizer.post_processor = qwen.post_processor
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        show_progress=False,
    )
    tokenizer.train_from_iterator([training_text], trainer=trainer)
    return tokenizer


def measure_test(out, name, tokenizer, ids, raw):
    counts = Counter(ids)
    token_ids = sorted(counts)
    remap = {old_id: new_id for new_id, old_id in enumerate(token_ids)}
    dense_ids = [remap[token] for token in ids]
    vocabulary = BytesIO()
    write_vocabulary(vocabulary, tokenizer, token_ids)
    lengths = lengths_from_counts(counts[token] for token in token_ids)
    path = out / f"{name}.huffman"
    block = write_bundle(path, vocabulary.getvalue(), lengths, {"test": dense_ids}, magic=TEST_MAGIC)["test"]
    with path.open("rb") as handle:
        words, blocks = read_bundle(handle)
        [(split, restored_ids)] = blocks
        if split != "test" or restored_ids != dense_ids or b"".join(words[token] for token in restored_ids) != raw:
            raise ValueError(f"lossless test failed for {name}")

    header_bytes = len(TEST_MAGIC) + BLOCK.size
    file_bytes = path.stat().st_size
    if file_bytes != header_bytes + vocabulary.tell() + len(lengths) + block["payload_bytes"]:
        raise ValueError(f"incorrect file byte breakdown for {name}")
    result = {
        "base_vocabulary_entries": tokenizer.get_vocab_size(with_added_tokens=False),
        "test_vocabulary_entries": len(token_ids),
        **block,
        "vocabulary_bytes": vocabulary.tell(),
        "code_table_bytes": len(lengths),
        "header_bytes": header_bytes,
        "file_bytes": file_bytes,
        "file_sha256": sha256(path.read_bytes()).hexdigest(),
        "lossless": True,
    }
    print(name, result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "qwen", "source", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()

    process = psutil.Process()
    process.nice(psutil.IDLE_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:1])
    import tokenizers

    started = perf_counter()
    if sha256(args.archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("archive differs from the previous WikiText-2 raw release")
    previous = json.loads((args.source / "report.json").read_bytes())
    qwen_digest = sha256(args.qwen.read_bytes()).hexdigest()
    if qwen_digest != previous["tokenizer_sha256"]:
        raise ValueError("Qwen tokenizer differs from the previous run")
    with ZipFile(args.archive) as archive:
        train_raw = archive.read("wikitext-2-raw/wiki.train.raw")
        test_raw = archive.read("wikitext-2-raw/wiki.test.raw")
    qwen = tokenizers.Tokenizer.from_file(str(args.qwen))
    test_text = test_raw.decode("utf-8")
    qwen_ids = qwen.encode(test_text, add_special_tokens=False).ids
    test_ids_file = (args.source / "test.ids").read_bytes()
    symbols = max(qwen.get_vocab().values()) + 1
    with BytesIO(test_ids_file) as handle:
        if (sha256(test_ids_file).hexdigest() != previous["splits"]["test"]["id_sha256"]
                or qwen_ids != read_tokens(handle, symbols) or handle.read(1)):
            raise ValueError("Qwen test sequence differs from the previous run")
    args.out.mkdir(parents=True, exist_ok=False)
    results = {"qwen3": measure_test(args.out, "qwen3", qwen, qwen_ids, test_raw)}

    training_text = train_raw.decode("utf-8")
    for vocab_size in (32_768, qwen.get_vocab_size(with_added_tokens=False)):
        name = f"wikitext-{vocab_size}"
        train_started = perf_counter()
        local = train_tokenizer(qwen, training_text, vocab_size)
        train_seconds = perf_counter() - train_started
        saved = args.out / f"{name}-tokenizer.json"
        local.save(str(saved))
        ids = local.encode(test_text, add_special_tokens=False).ids
        result = measure_test(args.out, name, local, ids, test_raw)
        result.update(vocabulary_target=vocab_size, training_seconds=train_seconds,
                      tokenizer_sha256=sha256(saved.read_bytes()).hexdigest())
        results[name] = result

    report = {
        "archive_sha256": ARCHIVE_SHA256,
        "qwen_tokenizer_sha256": qwen_digest,
        "qwen_test_ids_sha256": sha256(test_ids_file).hexdigest(),
        "train_bytes": len(train_raw),
        "train_sha256": sha256(train_raw).hexdigest(),
        "test_bytes": len(test_raw),
        "test_sha256": sha256(test_raw).hexdigest(),
        "python": sys.version.split()[0],
        "tokenizers": tokenizers.__version__,
        "priority": int(process.nice()),
        "cpu_affinity": process.cpu_affinity(),
        "seconds": perf_counter() - started,
        "results": results,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
