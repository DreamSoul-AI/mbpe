"""Comparison mergers used beside Tokenizer.merge_pair. They do not replace it."""

import math

import torch

from .patch import Patchify
from .state import State
from .tokenizer import TensorEncoding, Tokenizer

ALPHA = 0.3
SIGMA = 2.0


def first_index(scores):
    best = 0
    for index, score in enumerate(scores):
        if score > scores[best]:
            best = index
    return best


def spatial_consistency(offsets):
    count = len(offsets)
    mean_row = sum(row for row, _ in offsets) / count
    mean_col = sum(col for _, col in offsets) / count
    total = 0.0
    for row, col in offsets:
        distance = (row - mean_row) ** 2 + (col - mean_col) ** 2
        total += math.exp(-distance / (2 * SIGMA ** 2))
    return total / count


def priority(count, total, offsets):
    return count / total + ALPHA * spatial_consistency(offsets)


class GridMerger:
    """Horizontal and vertical pair merger. Selection is count, ratio, or priority."""

    def __init__(self, grid, score, new_symbol_limit=None, trace=False):
        self.height = len(grid)
        self.width = len(grid[0])
        self.tokens = {}
        self.owner = {}
        self.next_id = 0
        self.chosen = []
        self.merge_symbols = set()
        self.new_symbol_limit = new_symbol_limit
        self.trace = [] if trace else None
        for row, line in enumerate(grid):
            for col, value in enumerate(line):
                self._add(value, {(row, col): value})
        self._run(score)

    def _add(self, symbol, cells):
        token_id = self.next_id
        self.next_id += 1
        self.tokens[token_id] = {"symbol": symbol, "cells": dict(cells)}
        for cell in cells:
            self.owner[cell] = token_id
        return token_id

    def _occurrences(self):
        found = {}
        for row in range(self.height):
            for col in range(self.width):
                left = self.owner.get((row, col))
                for direction, neighbor in (
                    ((0, 1), (row, col + 1)),
                    ((1, 0), (row + 1, col)),
                ):
                    right = self.owner.get(neighbor)
                    if right is None or right == left:
                        continue
                    pair = (self.tokens[left]["symbol"], self.tokens[right]["symbol"])
                    bucket = found.setdefault(pair, [])
                    occurrence = (left, right, direction, (row, col))
                    if occurrence not in bucket:
                        bucket.append(occurrence)
        return found

    def _select(self, found, score):
        items = [(pair, places) for pair, places in found.items() if len(places) >= 2]
        if not items:
            return None
        items.sort(key=lambda item: min(place[3] for place in item[1]))
        counts = [len(places) for _, places in items]
        total = sum(counts)
        if score == "count":
            values = counts
        elif score == "ratio":
            values = [count / total for count in counts]
        elif score == "priority":
            values = [
                priority(len(places), total, [place[2] for place in places])
                for _, places in items
            ]
        else:
            raise ValueError(score)
        pair, places = items[first_index(values)]
        return pair, places

    def _run(self, score):
        while len(self.tokens) > 1:
            if self.new_symbol_limit is not None and len(self.merge_symbols) >= self.new_symbol_limit:
                break
            found = self._occurrences()
            selected = self._select(found, score)
            if selected is None:
                break
            pair, places = selected
            if self.trace is not None:
                self.trace.append({
                    "pair": pair,
                    "token_count": len(self.tokens),
                    "annotation": _annotate_pair(pair, places, found),
                })
            self.chosen.append(pair)
            consumed = set()
            merged = 0
            for left, right, direction, _anchor in sorted(places, key=lambda item: item[3]):
                if left in consumed or right in consumed or left not in self.tokens or right not in self.tokens:
                    continue
                cells = {}
                cells.update(self.tokens[left]["cells"])
                cells.update(self.tokens[right]["cells"])
                for token_id in (left, right):
                    for cell in self.tokens[token_id]["cells"]:
                        del self.owner[cell]
                    del self.tokens[token_id]
                symbol = ("m", pair[0], pair[1], direction)
                self.merge_symbols.add(symbol)
                self._add(symbol, cells)
                consumed.add(left)
                consumed.add(right)
                merged += 1
            if merged == 0:
                break

    def values(self):
        painted = {}
        for token in self.tokens.values():
            painted.update(token["cells"])
        return [[painted[(row, col)] for col in range(self.width)] for row in range(self.height)]


def _annotate_pair(pair, places, found):
    repeated = [item for item in found.values() if len(item) >= 2]
    total = sum(len(item) for item in repeated)
    count = len(places)
    offsets = [place[2] for place in places]
    directions = set(offsets)
    if directions == {(0, 1)}:
        direction = "水平"
    elif directions == {(1, 0)}:
        direction = "垂直"
    else:
        direction = "深层"
    return {
        "members": ["原始" if isinstance(symbol, int) else "合成" for symbol in pair],
        "direction": direction,
        "count_score": count,
        "spatial_score": priority(count, total, offsets),
    }


def first_fork(grid, new_symbol_limit=None):
    """First step where count and the published spatial score choose different pairs."""

    by_count = GridMerger(grid, "count", new_symbol_limit, trace=True)
    by_spatial = GridMerger(grid, "priority", new_symbol_limit, trace=True)
    shared = min(len(by_count.trace), len(by_spatial.trace))
    fork = None
    for index in range(shared):
        count_step = by_count.trace[index]
        spatial_step = by_spatial.trace[index]
        if count_step["pair"] != spatial_step["pair"]:
            fork = {
                "step": index,
                "token_count": count_step["token_count"],
                "spatial_token_count": spatial_step["token_count"],
                "count_pair": count_step["pair"],
                "spatial_pair": spatial_step["pair"],
                "count_annotation": count_step["annotation"],
                "spatial_annotation": spatial_step["annotation"],
            }
            break
    if fork is None and len(by_count.trace) != len(by_spatial.trace):
        longer = by_count if len(by_count.trace) > len(by_spatial.trace) else by_spatial
        step = longer.trace[shared]
        fork = {
            "step": shared,
            "token_count": step["token_count"],
            "spatial_token_count": step["token_count"],
            "count_pair": step["pair"] if longer is by_count else None,
            "spatial_pair": step["pair"] if longer is by_spatial else None,
            "count_annotation": step["annotation"] if longer is by_count else None,
            "spatial_annotation": step["annotation"] if longer is by_spatial else None,
        }
    return {
        "fork": fork,
        "count_final_tokens": len(by_count.tokens),
        "spatial_final_tokens": len(by_spatial.tokens),
        "count_lossless": by_count.values() == grid,
        "spatial_lossless": by_spatial.values() == grid,
        "count_vocab_symbols": len(by_count.merge_symbols),
        "spatial_vocab_symbols": len(by_spatial.merge_symbols),
    }


def merge_sequence(values, score, new_symbol_limit=None):
    """Merge a patch-order sequence until no pair occurs twice."""

    tokens = [{"symbol": value, "cells": [index]} for index, value in enumerate(values)]
    chosen = []
    merge_symbols = set()
    while len(tokens) > 1:
        if new_symbol_limit is not None and len(merge_symbols) >= new_symbol_limit:
            break
        counts = {}
        places = {}
        for index in range(len(tokens) - 1):
            pair = (tokens[index]["symbol"], tokens[index + 1]["symbol"])
            counts[pair] = counts.get(pair, 0) + 1
            places.setdefault(pair, []).append(index)
        repeated = [pair for pair, count in counts.items() if count >= 2]
        if not repeated:
            break
        repeated.sort(key=lambda pair: places[pair][0])
        total = sum(counts[pair] for pair in repeated)
        if score == "count":
            ranked = [counts[pair] for pair in repeated]
        elif score == "ratio":
            ranked = [counts[pair] / total for pair in repeated]
        else:
            raise ValueError(score)
        pair = repeated[first_index(ranked)]
        chosen.append(pair)
        merge_symbols.add(("m", pair[0], pair[1]))
        consumed = set()
        merged = []
        index = 0
        while index < len(tokens):
            if (
                index not in consumed
                and index + 1 < len(tokens)
                and index + 1 not in consumed
                and (tokens[index]["symbol"], tokens[index + 1]["symbol"]) == pair
            ):
                merged.append({
                    "symbol": ("m", pair[0], pair[1]),
                    "cells": tokens[index]["cells"] + tokens[index + 1]["cells"],
                })
                consumed.add(index)
                consumed.add(index + 1)
                index += 2
            else:
                merged.append(tokens[index])
                index += 1
        if len(merged) == len(tokens):
            break
        tokens = merged
    return tokens, chosen, merge_symbols


def patch_order(height, width):
    tokenizer = Tokenizer(
        {"root": 0.0, "merge": 0.0, "freq_counter": 0.0},
        max_codeword_size=(1, 2, 2),
        batch_size=1,
        max_workers=0,
    )
    data = torch.arange(height * width, dtype=torch.int64).reshape(1, 1, height, width)
    encoded = tokenizer.encode(data)
    order = []
    for token in encoded.sequences[0]:
        _flatten_index(token, order)
    return order


def _flatten_index(token, order):
    if isinstance(token, tuple):
        for item in token:
            _flatten_index(item, order)
    else:
        order.append(int(token))


def sequence_leaves(symbol):
    if isinstance(symbol, tuple) and symbol and symbol[0] == "m" and len(symbol) == 3:
        return sequence_leaves(symbol[1]) + sequence_leaves(symbol[2])
    return [symbol]


def encode_skipping_unknown(tokenizer, data):
    """Inference that keeps merging the next known pair when the maximum is unknown."""

    tokenizer._validate_input(data)
    vocab_size = len(tokenizer.vocab)
    frequency_total = tokenizer.freq_counter.total_count
    sequences = []
    for index in range(int(data.shape[0])):
        state = State(data[index:index + 1])
        for step, codeword_size in enumerate(tokenizer.codeword_sizes):
            state = tokenizer.patchify(step, state, Patchify(codeword_size))
            state = tokenizer.make_root(step, state, codeword_size, update=False)
            state = _merge_known_pair(tokenizer, step, state, codeword_size)
        sequences.append(tuple(state.joined))
    if len(tokenizer.vocab) != vocab_size or tokenizer.freq_counter.total_count != frequency_total:
        raise RuntimeError("skip encode must not modify the vocabulary or frequency counts")
    return TensorEncoding(tuple(sequences), tuple(int(dim) for dim in data.shape), data.dtype)


def _merge_known_pair(tokenizer, step, state, codeword_size):
    while True:
        pairs = state.pair(state.joined)
        if not pairs:
            break
        if tokenizer.freq_counter.total_count == 0:
            break
        freqs = tokenizer.freq_counter.get_freqs(pairs)
        order = sorted(range(len(pairs)), key=lambda item: (-freqs[item], item))
        codeword = None
        chosen = None
        for item in order:
            pair = pairs[item]
            codeword = tokenizer.vocab.get_codeword(pair, codeword_size)
            if codeword is not None:
                chosen = pair
                break
        if codeword is None:
            break
        state.update(joined=state.merge(state.joined, chosen, codeword))
    if step == len(tokenizer.codeword_sizes) - 1:
        state.finalize()
    return state
