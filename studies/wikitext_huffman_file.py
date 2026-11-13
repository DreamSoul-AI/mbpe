"""Store the frozen WikiText-2 BPE tokens using canonical Huffman coding."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
from itertools import chain
import json
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZipFile

import psutil

from bpe_block_file import BLOCK, read_exact
from canonical_huffman import CanonicalHuffman, lengths_from_counts
from wikitext_fixed_file import ARCHIVE_SHA256, SPLITS, read_tokens, read_vocabulary


MAGIC = b"WTH1"
TEST_MAGIC = b"WTH2"
FORMATS = {MAGIC: SPLITS, TEST_MAGIC: ("test",)}


def write_bundle(path, vocabulary_bytes, lengths, sequences, *, magic=MAGIC):
    codec = CanonicalHuffman(lengths)
    blocks = {}
    with path.open("xb") as handle:
        handle.write(magic)
        handle.write(vocabulary_bytes)
        handle.write(lengths)
        for split in FORMATS[magic]:
            ids = sequences[split]
            payload, bits = codec.encode(ids)
            handle.write(BLOCK.pack(len(ids), bits))
            handle.write(payload)
            blocks[split] = {"tokens": len(ids), "payload_bits": bits, "payload_bytes": len(payload)}
    return blocks


def read_blocks(handle, codec, splits):
    for split in splits:
        count, bits = BLOCK.unpack(read_exact(handle, BLOCK.size))
        payload = read_exact(handle, (bits + 7) // 8)
        yield split, codec.decode(payload, bits, count)
    if handle.read(1):
        raise ValueError("trailing data in WikiText Huffman file")


def read_bundle(handle):
    magic = read_exact(handle, len(MAGIC))
    if magic not in FORMATS:
        raise ValueError("incorrect WikiText Huffman file signature")
    vocabulary = read_vocabulary(handle)
    codec = CanonicalHuffman(read_exact(handle, len(vocabulary)))
    return vocabulary, read_blocks(handle, codec, FORMATS[magic])


def measure(args, process):
    started = perf_counter()
    previous = json.loads((args.source / "report.json").read_bytes())
    fixed = (args.source / "wikitext-2.bpe").read_bytes()
    fixed_digest = sha256(fixed).hexdigest()
    if fixed_digest != previous["bundle"]["file_sha256"]:
        raise ValueError("fixed-width file differs from the previous run")
    zip_bytes = args.archive.read_bytes()
    if sha256(zip_bytes).hexdigest() != ARCHIVE_SHA256 or previous["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("archive differs from the previous WikiText-2 raw release")

    with BytesIO(fixed) as source:
        vocabulary = read_vocabulary(source)
        vocabulary_end = source.tell()
        sequences = {split: read_tokens(source, len(vocabulary)) for split in SPLITS}
        if source.read(1):
            raise ValueError("trailing data in fixed-width file")
    if any(len(ids) != previous["splits"][split]["tokens"] for split, ids in sequences.items()):
        raise ValueError("token counts differ from the previous run")
    counts = Counter(chain.from_iterable(sequences.values()))
    lengths = lengths_from_counts(counts[symbol] for symbol in range(len(vocabulary)))

    args.out.mkdir(parents=True, exist_ok=False)
    bundle = args.out / "wikitext-2.huffman"
    blocks = write_bundle(bundle, fixed[:vocabulary_end], lengths, sequences)
    with bundle.open("rb") as handle, ZipFile(BytesIO(zip_bytes)) as archive:
        restored_vocabulary, restored_blocks = read_bundle(handle)
        if restored_vocabulary != vocabulary:
            raise ValueError("Huffman file changed the vocabulary")
        for split, ids in restored_blocks:
            raw = archive.read(f"wikitext-2-raw/wiki.{split}.raw")
            raw_digest = sha256(raw).hexdigest()
            restored = b"".join(restored_vocabulary[token] for token in ids)
            if ids != sequences[split] or restored != raw or raw_digest != previous["splits"][split]["raw_sha256"]:
                raise ValueError(f"frozen tokens or raw bytes changed for {split}")
            blocks[split].update(raw_bytes=len(raw), raw_sha256=raw_digest, lossless=True)
            print(split, blocks[split], flush=True)

    header_bytes = len(MAGIC) + len(SPLITS) * BLOCK.size
    payload_bytes = sum(block["payload_bytes"] for block in blocks.values())
    total = bundle.stat().st_size
    if total != header_bytes + vocabulary_end + len(lengths) + payload_bytes:
        raise ValueError("file size differs from the sum of its parts")
    report = {
        "archive_sha256": ARCHIVE_SHA256,
        "zip_bytes": len(zip_bytes),
        "fixed_sha256": fixed_digest,
        "fixed_bytes": len(fixed),
        "huffman_sha256": sha256(bundle.read_bytes()).hexdigest(),
        "huffman_bytes": total,
        "vocabulary_bytes": vocabulary_end,
        "code_table_bytes": len(lengths),
        "header_bytes": header_bytes,
        "payload_bytes": payload_bytes,
        "splits": blocks,
        "python": sys.version.split()[0],
        "priority": int(process.nice()),
        "cpu_affinity": process.cpu_affinity(),
        "seconds": perf_counter() - started,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Huffman: {total:,} bytes; fixed: {len(fixed):,}; ZIP: {len(zip_bytes):,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("measure")
    for arg in ("source", "archive", "out"):
        run.add_argument(f"--{arg}", required=True, type=Path)
    decode = commands.add_parser("decode")
    decode.add_argument("bundle", type=Path)
    decode.add_argument("out", type=Path)
    args = parser.parse_args()

    process = psutil.Process()
    process.nice(psutil.IDLE_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:1])
    if args.command == "measure":
        measure(args, process)
    else:
        with args.bundle.open("rb") as handle:
            vocabulary, blocks = read_bundle(handle)
            args.out.mkdir(parents=True, exist_ok=False)
            for split, ids in blocks:
                (args.out / f"wiki.{split}.raw").write_bytes(b"".join(vocabulary[token] for token in ids))


if __name__ == "__main__":
    main()
