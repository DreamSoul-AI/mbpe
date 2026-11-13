"""Self-contained BPE archives for finite integer grids and sparse domains."""

import argparse
from dataclasses import dataclass
from itertools import product
import json
from math import prod
from pathlib import Path
import struct
import zlib

import psutil

from bpe_block_file import ZLIB_LEVEL, orders_for, pack_symbols, read_exact, unpack_symbols
from canonical_huffman import CanonicalHuffman, lengths_from_validation
from lossless_nd_candidate import LosslessNDCandidate, validate_domain


MAGIC = b"MBPG"
VERSION = 1
MODES = ("row", "column", "snake", "hilbert", "nd", "adaptive")
CODECS = ("zlib", "huffman")
HEADER = struct.Struct("<4sBBBBII")
BLOCK = struct.Struct("<BII")


@dataclass(frozen=True)
class Spec:
    mode: str
    rank: int
    base: tuple
    grammars: tuple

    def symbols(self, direction):
        return len(self.base) + len(self.grammars[direction])

    def validate(self):
        if self.mode not in MODES or not 0 <= self.rank <= 255:
            raise ValueError("invalid mode or rank")
        if self.mode in ("hilbert", "adaptive") and self.rank != 2:
            raise ValueError("this scan requires rank two")
        if len(self.grammars) != (2 if self.mode == "adaptive" else 1):
            raise ValueError("incorrect number of grammars")
        if any(type(value) is not int for value in self.base) or tuple(sorted(set(self.base))) != self.base:
            raise ValueError("the base alphabet must contain sorted, distinct integers")
        fields = 2 + (self.rank if self.mode == "nd" else 0)
        for rules in self.grammars:
            for token, rule in enumerate(rules, len(self.base)):
                if len(rule) != fields or any(type(value) is not int for value in rule):
                    raise ValueError("malformed merge rule")
                if any(not 0 <= child < token for child in rule[:2]):
                    raise ValueError("rule must reference earlier symbols")


@dataclass(frozen=True)
class Packet:
    shape: tuple
    domain: tuple | None
    literals: tuple
    tokens: tuple


def uint(value):
    if value < 0:
        raise ValueError("unsigned integer required")
    data = bytearray()
    while value >= 128:
        data.append((value & 127) | 128)
        value >>= 7
    data.append(value)
    return bytes(data)


def read_uint(handle):
    value, shift = 0, 0
    while True:
        digit = read_exact(handle, 1)[0]
        value |= (digit & 127) << shift
        if digit < 128:
            if shift and not digit:
                raise ValueError("noncanonical integer")
            return value
        shift += 7


def sint(value):
    return uint(2 * value if value >= 0 else -2 * value - 1)


def read_sint(handle):
    value = read_uint(handle)
    return (value >> 1) ^ -(value & 1)


def bit_width(symbols):
    return max(1, (max(1, symbols) - 1).bit_length())


def grid_points(shape):
    return product(*(range(n) for n in shape))


def scan_points(shape, domain, mode):
    if mode == "row":
        return list(grid_points(shape) if domain is None else domain)
    if mode == "column":
        return sorted(grid_points(shape) if domain is None else domain, key=lambda point: point[::-1])
    if mode == "hilbert":
        side, width = shape
        if side != width or side < 1 or side & (side - 1):
            raise ValueError("Hilbert requires a power-of-two square")
        points = orders_for(side)["hilbert"]
    elif not shape:
        points = [()]
    else:
        points = (
            (*prefix, i) for index, prefix in enumerate(grid_points(shape[:-1]))
            for i in (range(shape[-1] - 1, -1, -1) if index % 2 else range(shape[-1]))
        )
    if domain is None:
        return list(points)
    valid = set(domain)
    return [point for point in points if point in valid]


def packet_from_nd(packet):
    domain = packet["domain"]
    return Packet(tuple(packet["shape"]), None if domain is None else tuple(map(tuple, domain)),
                  tuple(value for _, value in packet["literals"]), tuple(packet["tokens"]))


def nd_decoder(spec):
    rules = [((left, right, tuple(delta)), token)
             for token, (left, right, *delta) in enumerate(spec.grammars[0], len(spec.base))]
    return LosslessNDCandidate.from_dictionary({"format": "nd-bpe-candidate-dictionary", "version": 1,
                                                "rank": spec.rank, "base": spec.base, "rules": rules})


def decode_packet(spec, direction, packet, model):
    if model is not None:
        return model.decode({"format": "nd-bpe-candidate-packet", "version": 1, "shape": packet.shape,
                             "domain": packet.domain, "tokens": packet.tokens,
                             "literals": list(enumerate(packet.literals, spec.symbols(direction)))})[1]
    mode = ("row", "column")[direction] if spec.mode == "adaptive" else spec.mode
    points = scan_points(packet.shape, packet.domain, mode)
    rules = spec.grammars[direction]
    symbols = spec.symbols(direction)
    cells = {}
    stack = list(reversed(packet.tokens))
    while stack:
        token = stack.pop()
        if len(spec.base) <= token < symbols:
            left, right = rules[token - len(spec.base)]
            stack.extend((right, left))
        else:
            if len(cells) == len(points):
                raise ValueError("decoded tokens overfill the domain")
            value = spec.base[token] if token < len(spec.base) else packet.literals[token - symbols]
            cells[points[len(cells)]] = value
    if len(cells) != len(points):
        raise ValueError("decoded tokens do not fill the domain")
    return cells


def grid_bytes(shape, domain, values):
    return (json.dumps([shape, domain, values], separators=(",", ":")) + "\n").encode("ascii")


def code_lengths(spec, directions, packets):
    tables = []
    for direction in range(len(spec.grammars)):
        symbols = spec.symbols(direction)
        sequences = [[min(token, symbols) for token in packet.tokens]
                     for marker, packet in zip(directions, packets, strict=True) if marker == direction]
        tables.append(lengths_from_validation(sequences, max(2, symbols + 1)))
    return tuple(tables)


def prepare_packet(spec, direction, packet):
    validate_domain(spec.rank, packet.shape, {})
    if packet.domain is not None:
        if packet.domain != tuple(sorted(set(packet.domain))):
            raise ValueError("invalid domain order")
        validate_domain(spec.rank, packet.shape, dict.fromkeys(packet.domain, 0))
    if (len(packet.literals) > len(packet.tokens) or any(type(value) is not int for value in packet.literals)
            or packet.literals != tuple(sorted(set(packet.literals))) or set(spec.base) & set(packet.literals)):
        raise ValueError("invalid literal table")
    cell_count = prod(packet.shape) if packet.domain is None else len(packet.domain)
    if len(packet.tokens) > cell_count or bool(cell_count) != bool(packet.tokens):
        raise ValueError("invalid number of tokens for the domain")
    if any(not 0 <= token < spec.symbols(direction) + len(packet.literals) for token in packet.tokens):
        raise ValueError("invalid token id")


def new_cost():
    return dict(header_bytes=0, alphabet_bytes=0, rule_bytes=0, geometry_bytes=0, code_table_bytes=0,
                direction_bytes=0, block_header_bytes=0, shape_bytes=0, domain_bytes=0, literal_bytes=0,
                payload_bytes=0, token_counts=[], payload_sizes=[], widths=[])


def finish_cost(cost):
    cost["total_bytes"] = sum(value for key, value in cost.items() if key.endswith("_bytes"))
    cost["tokens"] = sum(cost["token_counts"])
    return cost


def write_archive(handle, spec, directions, packets, codec, lengths=None, width=0):
    spec.validate()
    if not packets or len(directions) != len(packets) or codec not in CODECS or width < 0:
        raise ValueError("invalid archive configuration")
    if (codec == "huffman") != (lengths is not None) or (lengths is not None and width):
        raise ValueError("incorrect code table or bit width")
    if lengths is not None:
        if tuple(map(len, lengths)) != tuple(max(2, spec.symbols(d) + 1) for d in range(len(spec.grammars))):
            raise ValueError("incorrect Huffman alphabet")
        codes = tuple(CanonicalHuffman(table) for table in lengths)
    cost = new_cost()

    def emit(key, data):
        handle.write(data)
        cost[key] += len(data)

    emit("header_bytes", HEADER.pack(MAGIC, VERSION, CODECS.index(codec), spec.rank, MODES.index(spec.mode),
                                     len(packets), width))
    emit("alphabet_bytes", uint(len(spec.base)) + b"".join(map(sint, spec.base)))
    for direction, rules in enumerate(spec.grammars):
        symbols = spec.symbols(direction)
        emit("rule_bytes", uint(len(rules)) + pack_symbols((v for rule in rules for v in rule[:2]),
                                                         bit_width(symbols), symbols))
        emit("geometry_bytes", b"".join(sint(v) for rule in rules for v in rule[2:]))
    if lengths is not None:
        emit("code_table_bytes", b"".join(lengths))
    if spec.mode == "adaptive":
        emit("direction_bytes", pack_symbols(directions, 1, 2))
    elif any(directions):
        raise ValueError("fixed scans cannot carry directions")

    for direction, packet in zip(directions, packets, strict=True):
        prepare_packet(spec, direction, packet)
        symbols = spec.symbols(direction)
        if codec == "huffman":
            payload, size = codes[direction].encode(min(token, symbols) for token in packet.tokens)
            literal_ids = [token - symbols for token in packet.tokens if token >= symbols]
            if len(packet.literals) > 1:
                payload += pack_symbols(literal_ids, bit_width(len(packet.literals)), len(packet.literals))
        else:
            token_symbols = symbols + len(packet.literals)
            actual_width = width or bit_width(token_symbols)
            if actual_width < bit_width(token_symbols):
                raise ValueError("packing width is too narrow")
            payload = zlib.compress(pack_symbols(packet.tokens, actual_width, token_symbols), level=ZLIB_LEVEL)
            size = len(payload)
            cost["widths"].append(actual_width)
        sparse = packet.domain is not None
        emit("block_header_bytes", BLOCK.pack(sparse, len(packet.tokens), size))
        emit("shape_bytes", b"".join(map(uint, packet.shape)))
        emit("literal_bytes", uint(len(packet.literals)) + b"".join(map(sint, packet.literals)))
        if sparse:
            points = set(packet.domain)
            emit("domain_bytes", pack_symbols((point in points for point in grid_points(packet.shape)), 1, 2))
        emit("payload_bytes", payload)
        cost["payload_sizes"].append(len(payload))
        cost["token_counts"].append(len(packet.tokens))
    return finish_cost(cost)


def read_archive(handle, output=None):
    cost = new_cost()

    def take(key, size):
        cost[key] += size
        return read_exact(handle, size)

    def number(key, read):
        start = handle.tell()
        value = read(handle)
        cost[key] += handle.tell() - start
        return value

    magic, version, codec_id, rank, mode_id, count, width = HEADER.unpack(take("header_bytes", HEADER.size))
    if magic != MAGIC or version != VERSION or not count or codec_id >= len(CODECS) or mode_id >= len(MODES):
        raise ValueError("invalid grid archive header")
    mode, codec = MODES[mode_id], CODECS[codec_id]
    base = tuple(number("alphabet_bytes", read_sint) for _ in range(number("alphabet_bytes", read_uint)))
    grammars = []
    for _ in range(2 if mode == "adaptive" else 1):
        rules = number("rule_bytes", read_uint)
        symbols = len(base) + rules
        bits = bit_width(symbols)
        children = unpack_symbols(take("rule_bytes", (2 * rules * bits + 7) // 8), 2 * rules, bits, symbols)
        dimensions = rank if mode == "nd" else 0
        grammars.append(tuple((a, b, *(number("geometry_bytes", read_sint) for _ in range(dimensions)))
                              for a, b in zip(children[::2], children[1::2])))
    spec = Spec(mode, rank, base, tuple(grammars))
    spec.validate()
    lengths = None
    if codec == "huffman":
        if width:
            raise ValueError("unexpected width for Huffman archive")
        lengths = tuple(take("code_table_bytes", max(2, spec.symbols(d) + 1)) for d in range(len(grammars)))
        codes = tuple(CanonicalHuffman(table) for table in lengths)
    directions = (unpack_symbols(take("direction_bytes", (count + 7) // 8), count, 1, 2)
                  if mode == "adaptive" else [0] * count)
    model = nd_decoder(spec) if output is not None and mode == "nd" else None
    packets = []
    for direction in directions:
        sparse, tokens, size = BLOCK.unpack(take("block_header_bytes", BLOCK.size))
        shape = tuple(number("shape_bytes", read_uint) for _ in range(rank))
        literal_count = number("literal_bytes", read_uint)
        literals = tuple(number("literal_bytes", read_sint) for _ in range(literal_count))
        if sparse > 1 or tokens > prod(shape) or literal_count > tokens:
            raise ValueError("invalid grid block header")
        domain = None
        if sparse:
            volume = prod(shape)
            mask = unpack_symbols(take("domain_bytes", (volume + 7) // 8), volume, 1, 2)
            domain = tuple(point for point, present in zip(grid_points(shape), mask) if present)
        start = cost["payload_bytes"]
        symbols = spec.symbols(direction)
        if codec == "huffman":
            code = codes[direction]
            if not tokens <= size <= tokens * code.max_length:
                raise ValueError("invalid Huffman payload length")
            values = code.decode(take("payload_bytes", (size + 7) // 8), size, tokens)
            if any(value > symbols for value in values):
                raise ValueError("invalid Huffman token id")
            escapes = values.count(symbols)
            if escapes and not literal_count:
                raise ValueError("missing literal table")
            if literal_count > 1:
                bits = bit_width(literal_count)
                indexes = unpack_symbols(take("payload_bytes", (escapes * bits + 7) // 8), escapes, bits, literal_count)
                ids = iter(indexes)
                values = [symbols + next(ids) if value == symbols else value for value in values]
        else:
            token_symbols = symbols + literal_count
            bits = width or bit_width(token_symbols)
            expected = (tokens * bits + 7) // 8
            if bits < bit_width(token_symbols) or not 1 <= size <= 2 * expected + 32:
                raise ValueError("invalid zlib payload length")
            decoder = zlib.decompressobj()
            packed = decoder.decompress(take("payload_bytes", size), expected + 1)
            if len(packed) != expected or not decoder.eof or decoder.unused_data:
                raise ValueError("invalid zlib payload")
            values = unpack_symbols(packed, tokens, bits, token_symbols)
            cost["widths"].append(bits)
        packet = Packet(shape, domain, literals, tuple(values))
        prepare_packet(spec, direction, packet)
        if output is None:
            packets.append(packet)
        else:
            cells = decode_packet(spec, direction, packet, model)
            output.write(grid_bytes(shape, domain, [cells[point] for point in sorted(cells)]))
        cost["payload_sizes"].append(cost["payload_bytes"] - start)
        cost["token_counts"].append(tokens)
    if handle.read(1):
        raise ValueError("trailing bytes")
    return spec, directions, packets, codec, lengths, width, finish_cost(cost)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:1])
    with args.archive.open("rb") as source, args.output.open("xb") as output:
        *_, cost = read_archive(source, output)
    print(json.dumps({"cost": cost, "cpu": process.cpu_affinity(), "priority": int(process.nice())}))


if __name__ == "__main__":
    main()
