"""Dense-grid files with a row or column BPE stream for each image."""

import argparse
import json
from pathlib import Path
import zlib

import psutil

from bpe_block_file import (
    BLOCK, HEADER, ZLIB_LEVEL, FileSpec, decode_linear, file_cost, huffman_cost,
    orders_for, pack_symbols, read_exact, read_zlib_block, unpack_symbols,
)
from canonical_huffman import CanonicalHuffman


MAGIC = b"MBPA"
VERSION = 1
MODES = ("row", "column")
ZLIB, HUFFMAN = 0, 1


def archive_cost(spec, rule_bytes, directions, sizes, counts, lengths, bits):
    cost = file_cost(rule_bytes, 0, sizes, counts, spec)
    cost.update(
        mode="adaptive", rules_per_direction=len(spec.rules), actual_rules=2 * len(spec.rules),
        direction_bits=len(directions), direction_bytes=(len(directions) + 7) // 8,
        code_table_bytes=0,
    )
    cost["total_bytes"] += cost["direction_bytes"]
    if lengths is not None:
        huffman_cost(cost, b"".join(lengths), bits)
    return cost


def write_archive(handle, specs, directions, sequences, lengths=None):
    """Write both grammars, optional code lengths, packed directions, then blocks."""
    if tuple(spec.mode for spec in specs) != MODES:
        raise ValueError("expected row and column grammars, in that order")
    row, column = specs
    for spec in specs:
        spec.validate()
    if (row.side, row.alphabet, len(row.rules)) != (column.side, column.alphabet, len(column.rules)):
        raise ValueError("incompatible row and column grammars")
    if not sequences or len(directions) != len(sequences):
        raise ValueError("one direction is required for each block")
    markers = pack_symbols(directions, 1, len(MODES))
    codec = ZLIB
    if lengths is not None:
        if len(lengths) != len(MODES) or any(len(table) != row.symbols for table in lengths):
            raise ValueError("mismatched Huffman alphabets")
        codes = tuple(CanonicalHuffman(table) for table in lengths)
        codec = HUFFMAN

    handle.write(HEADER.pack(MAGIC, VERSION, codec, row.alphabet, row.side, len(row.rules), len(sequences)))
    rule_bytes = 0
    for spec in specs:
        children = (child for pair in spec.rules for child in pair)
        packed = pack_symbols(children, spec.width, spec.symbols)
        handle.write(packed)
        rule_bytes += len(packed)
    if lengths is not None:
        for table in lengths:
            handle.write(table)
    handle.write(markers)

    sizes, counts, bits = [], [], []
    for direction, values in zip(directions, sequences):
        count = len(values)
        if not 1 <= count <= row.cells:
            raise ValueError("invalid block token count")
        if codec == HUFFMAN:
            payload, size = codes[direction].encode(values)
            bits.append(size)
        else:
            payload = zlib.compress(pack_symbols(values, row.width, row.symbols), level=ZLIB_LEVEL)
            size = len(payload)
        handle.write(BLOCK.pack(count, size))
        handle.write(payload)
        sizes.append(len(payload))
        counts.append(count)
    return archive_cost(row, rule_bytes, directions, sizes, counts, lengths, bits)


def scan_archive(handle, consume):
    magic, version, codec, alphabet, side, rules, samples = HEADER.unpack(read_exact(handle, HEADER.size))
    if magic != MAGIC or version != VERSION or codec not in (ZLIB, HUFFMAN):
        raise ValueError("unsupported adaptive archive format")
    FileSpec("row", side, alphabet).validate()
    if rules > 256 or not samples:
        raise ValueError("invalid rule or sample count")
    symbols = alphabet + rules
    width = (symbols - 1).bit_length()
    bytes_per_grammar = (2 * rules * width + 7) // 8
    specs = []
    for mode in MODES:
        children = unpack_symbols(read_exact(handle, bytes_per_grammar), 2 * rules, width, symbols)
        spec = FileSpec(mode, side, alphabet, tuple(zip(children[::2], children[1::2])))
        spec.validate()
        specs.append(spec)
    lengths = None
    if codec == HUFFMAN:
        lengths = tuple(read_exact(handle, symbols) for _ in MODES)
        codes = tuple(CanonicalHuffman(table) for table in lengths)
    directions = unpack_symbols(read_exact(handle, (samples + 7) // 8), samples, 1, len(MODES))

    sizes, counts, bits = [], [], []
    for direction in directions:
        spec = specs[direction]
        count, size = BLOCK.unpack(read_exact(handle, BLOCK.size))
        if not 1 <= count <= spec.cells:
            raise ValueError("invalid block token count")
        if codec == HUFFMAN:
            code = codes[direction]
            if not count <= size <= count * code.max_length:
                raise ValueError("invalid Huffman block size")
            payload_size = (size + 7) // 8
            tokens = code.decode(read_exact(handle, payload_size), size, count)
            bits.append(size)
        else:
            payload_size = size
            tokens = read_zlib_block(handle, count, size, spec)
        sizes.append(payload_size)
        counts.append(count)
        consume(spec, tokens)
    if handle.read(1):
        raise ValueError("trailing archive data")
    cost = archive_cost(specs[0], len(MODES) * bytes_per_grammar, directions, sizes, counts, lengths, bits)
    return tuple(specs), lengths, directions, cost


def read_token_archive(handle):
    sequences = []
    specs, lengths, directions, cost = scan_archive(handle, lambda spec, tokens: sequences.append(tokens))
    return specs, lengths, directions, sequences, cost


def read_archive(handle, output):
    def write_grid(spec, tokens):
        order = orders_for(spec.side)[spec.mode]
        output.write(decode_linear(tokens, spec, order))

    _, _, _, cost = scan_archive(handle, write_grid)
    return cost


def main():
    parser = argparse.ArgumentParser(description="Decode an adaptive BPE archive to row-major uint8 grids.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("raw_out", type=Path)
    args = parser.parse_args()
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:1])
    with args.archive.open("rb") as source, args.raw_out.open("xb") as output:
        cost = read_archive(source, output)
    memory = process.memory_info()
    cost.update(priority=int(process.nice()), cpu_affinity=process.cpu_affinity(),
                peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss))
    print(json.dumps(cost))


if __name__ == "__main__":
    main()
