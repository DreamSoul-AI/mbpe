"""Measure fixed-width files using the saved Elsner MNIST BPE rules."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
from struct import Struct, unpack
import subprocess
import sys
from time import perf_counter

import psutil

from bpe_block_file import pack_symbols, read_exact, unpack_symbols
from elsner_rule_decode import decode_sequence, rule_shapes


HEADER = Struct("<4sBBBBHI")
RULE = Struct("<HHbb")
ENTRY = Struct("<HBB")
COUNT = Struct("<H")
MAGIC = b"MBPF"
VERSION = 1
ALPHABET = 256
FORMATS = {"uint16-anchors": 0, "packed-anchors": 1, "packed-inferred": 2}
REPLAY = Path(__file__).with_name("elsner_apply_rules.py")


def init_cpu():
    process = psutil.Process()
    process.nice(psutil.IDLE_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:1])
    return process.cpu_affinity()


def anchor_width(shape):
    return (shape[0] - 1).bit_length() + (shape[1] - 1).bit_length()


def infer_positions(tokens, shapes, shape):
    height, width = shape
    filled = bytearray(height * width)
    cursor = 0
    for token in tokens:
        cursor = filled.find(0, cursor)
        if cursor == -1:
            raise ValueError("no remaining anchor")
        row, col = divmod(cursor, width)
        for dx, dy in shapes[token]:
            x, y = row + dx, col + dy
            if not (0 <= x < height and 0 <= y < width) or filled[x * width + y]:
                raise ValueError("invalid rule footprint")
            filled[x * width + y] = 1
        yield token, row, col


def encode_block(sequence, mode, shape, symbols):
    if mode == "uint16-anchors":
        return b"".join(ENTRY.pack(token, row, col) for token, row, col in sequence)
    width = (symbols - 1).bit_length()
    if mode == "packed-inferred":
        return pack_symbols((token for token, _, _ in sequence), width, symbols)
    y_bits = (shape[1] - 1).bit_length()
    coordinates = anchor_width(shape)
    values = ((token << coordinates) | (row << y_bits) | col for token, row, col in sequence)
    return pack_symbols(values, width + coordinates, symbols << coordinates)


def read_block(handle, mode, shape, symbols, shapes, count):
    if mode == "uint16-anchors":
        return list(ENTRY.iter_unpack(read_exact(handle, count * ENTRY.size)))
    width = (symbols - 1).bit_length()
    coordinates = 0 if mode == "packed-inferred" else anchor_width(shape)
    n_bits = width + coordinates
    values = unpack_symbols(read_exact(handle, (count * n_bits + 7) // 8), count, n_bits, symbols << coordinates)
    if mode == "packed-inferred":
        return list(infer_positions(values, shapes, shape))
    y_bits = (shape[1] - 1).bit_length()
    x_mask, y_mask = (1 << (coordinates - y_bits)) - 1, (1 << y_bits) - 1
    return [(value >> coordinates, (value >> y_bits) & x_mask, value & y_mask) for value in values]


def decode_file(path, output):
    with path.open("rb") as source, output.open("xb") as raw:
        magic, version, mode_id, height, width, merges, count = HEADER.unpack(read_exact(source, HEADER.size))
        mode = next((name for name, number in FORMATS.items() if number == mode_id), None)
        if magic != MAGIC or version != VERSION or mode is None or (height, width) != (28, 28) or merges != 256:
            raise ValueError("invalid MNIST BPE file header")
        rules = list(RULE.iter_unpack(read_exact(source, merges * RULE.size)))
        shapes = rule_shapes(rules)
        tokens = 0
        for _ in range(count):
            length, = COUNT.unpack(read_exact(source, COUNT.size))
            if not 1 <= length <= height * width:
                raise ValueError("invalid token count")
            seq = read_block(source, mode, (height, width), ALPHABET + merges, shapes, length)
            pixels = decode_sequence(seq, shapes, (height, width))
            raw.write(bytes(v for row in pixels for v in row))
            tokens += length
        if source.read(1):
            raise ValueError("trailing data")
    return {"images": count, "tokens": tokens}


def frozen_replay(torch, np):
    # Load the existing function bodies without their module-level thread setters.
    names = {"initial_state", "apply_one_rule", "apply_rules"}
    tree = ast.parse(REPLAY.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("missing rule replay functions")
    scope = {"torch": torch, "numpy": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REPLAY), "exec"), scope)
    return scope["apply_rules"]


def triplets(unique, classes, np):
    flat = unique.numpy().reshape(-1)
    first = np.sort(np.unique(flat, return_index=True)[1])
    values = classes.numpy().reshape(-1)
    return [(int(values[i]), *divmod(int(i), unique.shape[1])) for i in first]


def prepare(args, affinity):
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    blob = args.idx.read_bytes()
    if unpack(">IIII", blob[:16]) != (2051, 10_000, 28, 28) or len(blob) != 16 + 10_000 * 784:
        raise ValueError("expected the original MNIST test image file")
    pixels = blob[16:]
    raw = torch.from_numpy(np.frombuffer(pixels, dtype=np.uint8).copy().reshape(-1, 28, 28)).long()
    if not torch.equal(raw.float().div(255).mul(255).int(), raw):
        raise ValueError("the notebook normalization changes some pixel values")
    rules = [tuple(map(int, rule)) for rule in torch.load(args.rules, weights_only=True, map_location="cpu")]
    if len(rules) != 256:
        raise ValueError("expected 256 saved rules")
    expected = json.loads(args.summary.read_bytes())["test_full"]
    apply = frozen_replay(torch, np)
    shapes = rule_shapes(rules)
    counts = []
    first = []
    for path in sorted((args.rules.parent / "transcribed_data_og").glob("*/*.dat"))[:3]:
        relative = path.relative_to(args.rules.parent / "transcribed_data_og")
        image = torch.load(path, weights_only=True, map_location="cpu").long()
        stored = torch.load(args.rules.parent / "transcribed_data" / relative, weights_only=True, map_location="cpu")
        actual = apply(image, rules)
        if not all(torch.equal(a, b) for a, b in zip(actual, stored, strict=True)):
            raise ValueError(f"rule replay differs from stored state: {relative}")
        first.append(relative.as_posix())

    from contextlib import ExitStack

    args.out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    rule_bytes = b"".join(RULE.pack(*rule) for rule in rules)
    with ExitStack() as stack:
        handles = {name: stack.enter_context((args.out / f"{name}.mbpf").open("xb")) for name in FORMATS}
        for name, handle in handles.items():
            handle.write(HEADER.pack(MAGIC, VERSION, FORMATS[name], 28, 28, len(rules), len(raw)))
            handle.write(rule_bytes)
        for index, image in enumerate(raw, 1):
            sequence = triplets(*apply(image, rules), np)
            inferred = list(infer_positions((token for token, _, _ in sequence), shapes, (28, 28)))
            if inferred != sequence:
                raise ValueError(f"implicit anchors differ on image {index - 1}")
            counts.append(len(sequence))
            for name, handle in handles.items():
                handle.write(COUNT.pack(len(sequence)))
                handle.write(encode_block(sequence, name, (28, 28), ALPHABET + len(rules)))
            if index % 2000 == 0:
                print(f"Wrote {index}/{len(raw)} images to all three files", flush=True)
    if (len(counts), sum(counts), min(counts), max(counts)) != (
        expected["images"], round(expected["mean_length"] * expected["images"]), expected["min_length"], expected["max_length"]
    ):
        raise ValueError("token lengths differ from the historical run")

    results = {}
    for name in FORMATS:
        path = args.out / f"{name}.mbpf"
        decoded = args.out / f"{name}.pixels"
        receipt = subprocess.run([sys.executable, "-B", "-X", "utf8", __file__, "decode", str(path), str(decoded)],
                                 check=True, capture_output=True, text=True, encoding="utf-8")
        recovered = decoded.read_bytes()
        state = json.loads(receipt.stdout)
        if recovered != pixels or state != {"images": len(raw), "tokens": sum(counts), "affinity": affinity,
                                             "priority": int(psutil.IDLE_PRIORITY_CLASS)}:
            raise ValueError(f"independent decode failed: {path}")
        size = path.stat().st_size
        results[name] = {"bytes": size, "bytes_saved_percent": 100 * (1 - size / len(pixels)),
                         "archive_sha256": sha256(path.read_bytes()).hexdigest(), "lossless_images": len(raw)}
        print(f"{name}: {size} bytes, {len(raw)} images restored", flush=True)
    return {"pixel_bytes": len(pixels), "image_count": len(raw), "tokens": sum(counts),
            "raw_sha256": sha256(blob).hexdigest(), "pixels_sha256": sha256(pixels).hexdigest(),
            "rule_source_sha256": sha256(args.rules.read_bytes()).hexdigest(),
            "replay_source_sha256": sha256(REPLAY.read_bytes()).hexdigest(), "checked_old_states": first,
            "header_bytes": HEADER.size, "rule_bytes": len(rule_bytes), "image_count_field_bytes": COUNT.size,
            "seconds": perf_counter() - started, "affinity": affinity, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure = subparsers.add_parser("measure")
    for field in ("idx", "rules", "summary", "out"):
        measure.add_argument(f"--{field}", type=Path, required=True)
    decode = subparsers.add_parser("decode")
    decode.add_argument("archive", type=Path)
    decode.add_argument("pixels", type=Path)
    args = parser.parse_args()
    affinity = init_cpu()
    if args.command == "decode":
        print(json.dumps({**decode_file(args.archive, args.pixels), "affinity": affinity,
                          "priority": int(psutil.Process().nice())}))
    else:
        result = prepare(args, affinity)
        with (args.out / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
