"""Measure fixed-width BPE storage on the original WikiText-2 UTF-8 files."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from struct import Struct
import sys
from time import perf_counter
from zipfile import ZipFile

import psutil

from bpe_block_file import pack_symbols, read_exact, unpack_symbols


SPLITS = ("train", "valid", "test")
ARCHIVE_SHA256 = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
TOKEN_HEADER = Struct("<IB")
VOCAB_COUNT = Struct("<I")
TOKEN_LENGTH = Struct("<H")


def write_tokens(handle, ids, count, symbols):
    width = max(1, (symbols - 1).bit_length())
    handle.write(TOKEN_HEADER.pack(count, width))
    handle.write(pack_symbols(ids, width, symbols))


def read_tokens(handle, symbols):
    count, width = TOKEN_HEADER.unpack(read_exact(handle, TOKEN_HEADER.size))
    if width != max(1, (symbols - 1).bit_length()):
        raise ValueError("incorrect token width")
    payload = read_exact(handle, (count * width + 7) // 8)
    return unpack_symbols(payload, count, width, symbols)


def bytelevel_inverse():
    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    mapping = {chr(value): value for value in direct}
    for index, value in enumerate(sorted(set(range(256)) - direct)):
        mapping[chr(256 + index)] = value
    return mapping


def write_vocabulary(handle, tokenizer, token_ids):
    byte_values = bytelevel_inverse()
    handle.write(VOCAB_COUNT.pack(len(token_ids)))
    for token_id in token_ids:
        token = tokenizer.id_to_token(token_id)
        # Match the ByteLevel decoder's literal UTF-8 fallback for added tokens.
        spelling = (bytes(byte_values[char] for char in token)
                    if all(char in byte_values for char in token) else token.encode("utf-8"))
        handle.write(TOKEN_LENGTH.pack(len(spelling)))
        handle.write(spelling)


def write_bundle(path, tokenizer, used_ids, symbols):
    remap = {old_id: new_id for new_id, old_id in enumerate(sorted(used_ids))}
    with path.open("xb") as bundle:
        write_vocabulary(bundle, tokenizer, remap)
        codebook_bytes = bundle.tell()
        sizes = {}
        for split in SPLITS:
            with path.with_name(f"{split}.ids").open("rb") as handle:
                ids = read_tokens(handle, symbols)
            start = bundle.tell()
            write_tokens(bundle, (remap[token] for token in ids), len(ids), len(remap))
            sizes[split] = bundle.tell() - start
    return {"file_bytes": path.stat().st_size, "codebook_bytes": codebook_bytes,
            "codebook_entries": len(remap), "id_width": max(1, (len(remap) - 1).bit_length()),
            "split_id_file_bytes": sizes, "file_sha256": sha256(path.read_bytes()).hexdigest()}


def read_vocabulary(handle):
    count, = VOCAB_COUNT.unpack(read_exact(handle, VOCAB_COUNT.size))
    vocabulary = []
    for _ in range(count):
        length, = TOKEN_LENGTH.unpack(read_exact(handle, TOKEN_LENGTH.size))
        vocabulary.append(read_exact(handle, length))
    return vocabulary


def decode_bundle(path):
    with path.open("rb") as bundle:
        vocabulary = read_vocabulary(bundle)
        for split in SPLITS:
            ids = read_tokens(bundle, len(vocabulary))
            yield split, b"".join(vocabulary[token] for token in ids)
        if bundle.read(1):
            raise ValueError("trailing data in WikiText bundle")


def measure(args, affinity):
    import tokenizers

    started = perf_counter()
    archive_digest = sha256(args.archive.read_bytes()).hexdigest()
    if archive_digest != ARCHIVE_SHA256:
        raise ValueError("archive does not match Salesforce's WikiText-2 raw release")
    tokenizer = tokenizers.Tokenizer.from_file(str(args.tokenizer))
    symbols = max(tokenizer.get_vocab().values()) + 1
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "archive_sha256": archive_digest,
        "tokenizer_sha256": sha256(args.tokenizer.read_bytes()).hexdigest(),
        "tokenizer_json_bytes": args.tokenizer.stat().st_size,
        "python": sys.version.split()[0],
        "tokenizers": tokenizers.__version__,
        "priority": int(psutil.Process().nice()),
        "cpu_affinity": affinity,
        "shared_id_width": (symbols - 1).bit_length(),
        "splits": {},
    }
    used_ids = set()
    bundle_path = args.out / "wikitext-2.bpe"
    with ZipFile(args.archive) as source:
        for split in SPLITS:
            raw = source.read(f"wikitext-2-raw/wiki.{split}.raw")
            ids = tokenizer.encode(raw.decode("utf-8"), add_special_tokens=False).ids
            used_ids.update(ids)
            path = args.out / f"{split}.ids"
            with path.open("xb") as handle:
                write_tokens(handle, ids, len(ids), symbols)
            with path.open("rb") as handle:
                restored = tokenizer.decode(read_tokens(handle, symbols), skip_special_tokens=False).encode("utf-8")
                if restored != raw or handle.read(1):
                    raise ValueError(f"shared vocabulary round trip failed on {split}")
            report["splits"][split] = {
                "raw_bytes": len(raw), "tokens": len(ids), "id_file_bytes": path.stat().st_size,
                "raw_sha256": sha256(raw).hexdigest(), "id_sha256": sha256(path.read_bytes()).hexdigest(),
                "shared_vocabulary_lossless": True,
            }
            print(split, report["splits"][split], flush=True)
        report["bundle"] = write_bundle(bundle_path, tokenizer, used_ids, symbols)
        for split, raw in decode_bundle(bundle_path):
            if raw != source.read(f"wikitext-2-raw/wiki.{split}.raw"):
                raise ValueError(f"bundle round trip failed on {split}")
            report["splits"][split]["bundle_lossless"] = True
    report["seconds"] = perf_counter() - started
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("bundle", report["bundle"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("measure")
    run.add_argument("--archive", required=True, type=Path)
    run.add_argument("--tokenizer", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    decode = commands.add_parser("decode")
    decode.add_argument("bundle", type=Path)
    decode.add_argument("out", type=Path)
    args = parser.parse_args()

    process = psutil.Process()
    process.nice(psutil.IDLE_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:1])
    if args.command == "measure":
        measure(args, process.cpu_affinity())
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        for split, data in decode_bundle(args.bundle):
            (args.out / f"wiki.{split}.raw").write_bytes(data)


if __name__ == "__main__":
    main()
