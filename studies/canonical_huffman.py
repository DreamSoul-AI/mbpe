"""Canonical, deterministic token coding for the block-file study."""

from collections import Counter
import heapq
from operator import index


def lengths_from_validation(sequences, symbols):
    """Build code lengths from development blocks, with one count per legal ID."""
    counts = Counter(value for values in sequences for value in values)
    if symbols < 2 or any(not 0 <= value < symbols for value in counts):
        raise ValueError("invalid Huffman alphabet")
    return lengths_from_counts(counts[symbol] + 1 for symbol in range(symbols))


def lengths_from_counts(counts):
    """Build Huffman code lengths from positive counts in symbol order."""
    weights = list(map(index, counts))
    if len(weights) < 2 or any(weight <= 0 for weight in weights):
        raise ValueError("expected positive counts for at least two symbols")
    heap = [(weight, symbol, (symbol,)) for symbol, weight in enumerate(weights)]
    heapq.heapify(heap)
    lengths = [0] * len(weights)
    while len(heap) > 1:
        first_weight, first_id, first_leaves = heapq.heappop(heap)
        second_weight, second_id, second_leaves = heapq.heappop(heap)
        leaves = first_leaves + second_leaves
        for symbol in leaves:
            lengths[symbol] += 1
        heapq.heappush(heap, (first_weight + second_weight, min(first_id, second_id), leaves))
    return bytes(lengths)


class CanonicalHuffman:
    def __init__(self, lengths):
        if len(lengths) < 2 or any(not 1 <= length < len(lengths) for length in lengths):
            raise ValueError("invalid Huffman code length")
        order = sorted(enumerate(lengths), key=lambda item: (item[1], item[0]))
        self.codes = [None] * len(lengths)
        self.symbols = {}
        code = last = 0
        for symbol, length in order:
            code <<= length - last
            if code >= 1 << length:
                raise ValueError("oversubscribed Huffman table")
            self.codes[symbol] = (code, length)
            self.symbols[length, code] = symbol
            code += 1
            last = length
        if code != 1 << last:
            raise ValueError("incomplete Huffman table")
        self.max_length = last

    def encode(self, values):
        data = bytearray()
        buffer = buffered = bit_count = 0
        for value in map(index, values):
            if not 0 <= value < len(self.codes):
                raise ValueError("token outside Huffman alphabet")
            code, length = self.codes[value]
            buffer = (buffer << length) | code
            buffered += length
            bit_count += length
            while buffered >= 8:
                buffered -= 8
                data.append(buffer >> buffered)
                buffer &= (1 << buffered) - 1
        if buffered:
            data.append(buffer << (8 - buffered))
        return bytes(data), bit_count

    def decode(self, data, bit_count, count):
        if not count <= bit_count <= count * self.max_length:
            raise ValueError("invalid Huffman bit count")
        if len(data) != (bit_count + 7) // 8:
            raise ValueError("incorrect Huffman payload size")
        padding = -bit_count % 8
        if padding and data[-1] & ((1 << padding) - 1):
            raise ValueError("nonzero Huffman padding")
        values = []
        code = length = 0
        for offset in range(bit_count):
            code = (code << 1) | ((data[offset // 8] >> (7 - offset % 8)) & 1)
            length += 1
            symbol = self.symbols.get((length, code))
            if symbol is not None:
                values.append(symbol)
                code = length = 0
            elif length == self.max_length:
                raise ValueError("invalid Huffman code")
        if length or len(values) != count:
            raise ValueError("incomplete Huffman code or incorrect token count")
        return values
