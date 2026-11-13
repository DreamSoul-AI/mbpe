"""Self-contained dense-grid files for BPE compression comparisons."""

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from operator import index
from pathlib import Path
import struct
import zlib

import psutil

from batch30_mask_baselines import make_orders
from canonical_huffman import CanonicalHuffman
from lossless_nd_candidate import LosslessNDCandidate


ORDERS = ("row", "column", "snake", "hilbert")
METHODS = ("raw", *ORDERS, "2d")
MAGIC = b"MBPE"
VERSION = 1
HUFFMAN_VERSION = 2
ZLIB_LEVEL = 9
HEADER = struct.Struct("<4sBBBBHI")
BLOCK = struct.Struct("<II")
DELTA = struct.Struct("<bb")


@dataclass(frozen=True)
class FileSpec:
    """One alphabet, square shape, traversal, and shared grammar per file."""

    mode: str
    side: int
    alphabet: int
    rules: tuple[tuple[int, ...], ...] = ()

    @property
    def cells(self):
        return self.side * self.side

    @property
    def symbols(self):
        return self.alphabet + len(self.rules)

    @property
    def width(self):
        return (self.symbols - 1).bit_length()

    def validate(self):
        if self.mode not in METHODS or self.alphabet not in (2, 4):
            raise ValueError("unsupported method or alphabet")
        if not 1 <= self.side <= 128 or self.side & (self.side - 1):
            raise ValueError("expected a power-of-two square with side at most 128")
        if len(self.rules) > 256 or (self.mode == "raw" and self.rules):
            raise ValueError("invalid rule count")
        expected_fields = 4 if self.mode == "2d" else 2
        areas = [1] * self.alphabet
        for rule in self.rules:
            if len(rule) != expected_fields:
                raise ValueError("invalid rule size")
            left, right, *delta = rule
            if not (0 <= left < len(areas) and 0 <= right < len(areas)):
                raise ValueError("rule must reference earlier symbols")
            area = areas[left] + areas[right]
            if area > self.cells or any(abs(step) >= self.side for step in delta):
                raise ValueError("rule exceeds the grid")
            areas.append(area)


def pack_symbols(values, width, symbols):
    packed = bytearray()
    buffer = bits = 0
    for value in map(index, values):
        if not 0 <= value < symbols:
            raise ValueError("symbol outside the alphabet and grammar")
        buffer = (buffer << width) | value
        bits += width
        while bits >= 8:
            bits -= 8
            packed.append(buffer >> bits)
            buffer &= (1 << bits) - 1
    if bits:
        packed.append(buffer << (8 - bits))
    return bytes(packed)


def unpack_symbols(packed, count, width, symbols):
    if len(packed) != (count * width + 7) // 8:
        raise ValueError("incorrect packed length")
    values = []
    buffer = bits = 0
    for byte in packed:
        buffer = (buffer << 8) | byte
        bits += 8
        while bits >= width and len(values) < count:
            bits -= width
            value = buffer >> bits
            if value >= symbols:
                raise ValueError("packed symbol outside the alphabet and grammar")
            values.append(value)
            buffer &= (1 << bits) - 1
    if len(values) != count or buffer:
        raise ValueError("invalid packed padding or symbol count")
    return values


def file_cost(rule_bytes, geometry_bytes, payload_sizes, token_counts, spec):
    block_headers = BLOCK.size * len(payload_sizes)
    return {
        "header_bytes": HEADER.size,
        "rule_symbol_bytes": rule_bytes,
        "geometry_bytes": geometry_bytes,
        "block_header_bytes": block_headers,
        "payload_bytes": sum(payload_sizes),
        "total_bytes": HEADER.size + rule_bytes + geometry_bytes + block_headers + sum(payload_sizes),
        "payload_sizes": payload_sizes,
        "token_counts": token_counts,
        "actual_rules": len(spec.rules),
        "symbol_width": spec.width,
        "images": len(payload_sizes),
        "side": spec.side,
        "mode": spec.mode,
        "alphabet": spec.alphabet,
    }


def write_prefix(handle, spec, samples, version):
    handle.write(HEADER.pack(MAGIC, version, METHODS.index(spec.mode), spec.alphabet,
                             spec.side, len(spec.rules), samples))
    children = [child for rule in spec.rules for child in rule[:2]]
    rule_bytes = pack_symbols(children, spec.width, spec.symbols)
    geometry = b"".join(DELTA.pack(*rule[2:]) for rule in spec.rules) if spec.mode == "2d" else b""
    handle.write(rule_bytes)
    handle.write(geometry)
    return len(rule_bytes), len(geometry)


def write_archive(handle, spec, sequences):
    spec.validate()
    if not sequences:
        raise ValueError("empty archive")
    rule_size, geometry_size = write_prefix(handle, spec, len(sequences), VERSION)
    payload_sizes, token_counts = [], []
    for values in sequences:
        count = len(values)
        if not 1 <= count <= spec.cells or (spec.mode == "raw" and count != spec.cells):
            raise ValueError("invalid block token count")
        compressed = zlib.compress(pack_symbols(values, spec.width, spec.symbols), level=ZLIB_LEVEL)
        handle.write(BLOCK.pack(count, len(compressed)))
        handle.write(compressed)
        payload_sizes.append(len(compressed))
        token_counts.append(count)
    return file_cost(rule_size, geometry_size, payload_sizes, token_counts, spec)


def huffman_cost(cost, lengths, bits):
    cost.update(
        code_table_bytes=len(lengths), payload_bits=sum(bits),
        padding_bits=8 * cost["payload_bytes"] - sum(bits),
        total_bytes=cost["total_bytes"] + len(lengths),
    )
    return cost


def write_huffman_archive(handle, spec, sequences, lengths):
    spec.validate()
    if not sequences or len(lengths) != spec.symbols:
        raise ValueError("empty archive or mismatched Huffman alphabet")
    code = CanonicalHuffman(lengths)
    rule_size, geometry_size = write_prefix(handle, spec, len(sequences), HUFFMAN_VERSION)
    handle.write(lengths)
    sizes, counts, bits = [], [], []
    for values in sequences:
        count = len(values)
        if not 1 <= count <= spec.cells or (spec.mode == "raw" and count != spec.cells):
            raise ValueError("invalid block token count")
        payload, bit_count = code.encode(values)
        handle.write(BLOCK.pack(count, bit_count))
        handle.write(payload)
        sizes.append(len(payload))
        counts.append(count)
        bits.append(bit_count)
    return huffman_cost(file_cost(rule_size, geometry_size, sizes, counts, spec), lengths, bits)


def read_exact(handle, count):
    data = handle.read(count)
    if len(data) != count:
        raise ValueError("truncated archive")
    return data


@lru_cache(maxsize=8)
def orders_for(side):
    return make_orders(side)


def decode_linear(tokens, spec, order):
    flat = bytearray(spec.cells)
    stack = list(reversed(tokens))
    pos = 0
    while stack:
        token = stack.pop()
        if token < spec.alphabet:
            if pos == spec.cells:
                raise ValueError("sequence overfills the grid")
            y, x = order[pos]
            flat[y * spec.side + x] = token
            pos += 1
        else:
            left, right = spec.rules[token - spec.alphabet]
            stack.extend((right, left))
    if pos != spec.cells:
        raise ValueError("sequence does not fill the grid")
    return flat


def decode_geometry(tokens, spec, model):
    packet = {
        "format": "nd-bpe-candidate-packet", "version": 1, "shape": (spec.side, spec.side),
        "domain": None, "literals": [], "tokens": tokens,
    }
    _, cells = model.decode(packet)
    return bytes(cells[y, x] for y, x in orders_for(spec.side)["row"])


def read_prefix(handle):
    magic, version, method, alphabet, side, count, samples = HEADER.unpack(read_exact(handle, HEADER.size))
    if magic != MAGIC or version not in (VERSION, HUFFMAN_VERSION) or method >= len(METHODS):
        raise ValueError("unsupported archive format")
    if alphabet not in (2, 4) or count > 256 or not samples:
        raise ValueError("invalid alphabet, rule count, or sample count")
    mode = METHODS[method]
    symbols = alphabet + count
    width = (symbols - 1).bit_length()
    rule_bytes = read_exact(handle, (count * 2 * width + 7) // 8)
    children = unpack_symbols(rule_bytes, 2 * count, width, symbols)
    pairs = tuple(zip(children[::2], children[1::2]))
    geometry = read_exact(handle, count * DELTA.size) if mode == "2d" else b""
    rules = pairs
    if mode == "2d":
        rules = tuple((*pair, *delta) for pair, delta in zip(pairs, DELTA.iter_unpack(geometry)))
    spec = FileSpec(mode, side, alphabet, rules)
    spec.validate()
    return spec, samples, version, len(rule_bytes), len(geometry)


def scan_archive(handle, consumer):
    spec, samples, version, rule_size, geometry_size = read_prefix(handle)
    accept = consumer(spec)
    lengths = read_exact(handle, spec.symbols) if version == HUFFMAN_VERSION else None
    code = CanonicalHuffman(lengths) if lengths is not None else None
    payload_sizes, token_counts, bit_counts = [], [], []
    for _ in range(samples):
        count, size = BLOCK.unpack(read_exact(handle, BLOCK.size))
        if not 1 <= count <= spec.cells or (spec.mode == "raw" and count != spec.cells):
            raise ValueError("invalid block token count")
        if code is None:
            tokens = read_zlib_block(handle, count, size, spec)
            payload_sizes.append(size)
        else:
            if not count <= size <= count * code.max_length:
                raise ValueError("invalid Huffman block size")
            byte_count = (size + 7) // 8
            tokens = code.decode(read_exact(handle, byte_count), size, count)
            bit_counts.append(size)
            payload_sizes.append(byte_count)
        token_counts.append(count)
        accept(tokens)
    if handle.read(1):
        raise ValueError("trailing archive data")
    cost = file_cost(rule_size, geometry_size, payload_sizes, token_counts, spec)
    if lengths is not None:
        huffman_cost(cost, lengths, bit_counts)
    return spec, cost


def read_token_archive(handle):
    sequences = []
    spec, cost = scan_archive(handle, lambda _: sequences.append)
    return spec, sequences, cost


def read_zlib_block(handle, count, size, spec):
    expected_size = (count * spec.width + 7) // 8
    if not 1 <= size <= expected_size * 2 + 32:
        raise ValueError("invalid compressed block size")
    decoder = zlib.decompressobj()
    packed = decoder.decompress(read_exact(handle, size), expected_size + 1)
    if len(packed) != expected_size or not decoder.eof or decoder.unused_data:
        raise ValueError("invalid compressed block")
    return unpack_symbols(packed, count, spec.width, spec.symbols)


def read_archive(handle, output):
    def grid_consumer(spec):
        mode, side, alphabet, rules = spec.mode, spec.side, spec.alphabet, spec.rules
        if mode == "2d":
            model = LosslessNDCandidate.from_dictionary({
                "format": "nd-bpe-candidate-dictionary", "version": 1, "rank": 2,
                "base": list(range(alphabet)),
                "rules": [((left, right, (dy, dx)), alphabet + i)
                          for i, (left, right, dy, dx) in enumerate(rules)],
            })
            return lambda tokens: output.write(decode_geometry(tokens, spec, model))
        if mode == "raw":
            return lambda tokens: output.write(bytes(tokens))
        order = orders_for(side)[mode]
        return lambda tokens: output.write(decode_linear(tokens, spec, order))

    _, cost = scan_archive(handle, grid_consumer)
    return cost


def main():
    parser = argparse.ArgumentParser(description="Decode a BPE block archive to row-major uint8 grids.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("raw_out", type=Path)
    args = parser.parse_args()
    with args.archive.open("rb") as source, args.raw_out.open("xb") as output:
        result = read_archive(source, output)
    process = psutil.Process()
    memory = process.memory_info()
    result["peak_working_set_bytes"] = getattr(memory, "peak_wset", memory.rss)
    result["cpu_affinity"] = process.cpu_affinity()
    result["priority"] = int(process.nice())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
