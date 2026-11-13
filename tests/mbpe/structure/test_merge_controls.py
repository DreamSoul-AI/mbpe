"""Four synthetic-grid comparisons. They do not change Tokenizer.merge_pair."""

import os
import unittest

os.environ.setdefault("TQDM_DISABLE", "1")

import torch
from torch.utils.data import TensorDataset

from mbpe.structure.bpe.controls import GridMerger, first_fork, first_index as _first_index, merge_sequence as _merge_sequence
from mbpe.structure.bpe.controls import patch_order as _patch_order, priority as _priority
from mbpe.structure.bpe.state import State
from mbpe.structure.bpe.tokenizer import Tokenizer


def _tokenizer():
    return Tokenizer(
        {"root": 0.0, "merge": 0.0, "freq_counter": 0.0},
        max_codeword_size=(1, 2, 2),
        batch_size=1,
        max_workers=0,
    )


def _partition_from_merge(symbols, pair, codeword):
    state = State(torch.zeros(1, dtype=torch.uint8))
    state.joined = list(symbols)
    merged = state.merge(state.joined, pair, codeword)
    cells = []
    cursor = 0
    for token in merged:
        if token == codeword:
            cells.append((cursor, cursor + 1))
            cursor += 2
        else:
            cells.append((cursor,))
            cursor += 1
    return merged, cells


def _partition_from_ban(symbols, pair, order):
    occupied = [False] * len(symbols)
    spans = []
    for start in order:
        stop = start + 1
        if stop >= len(symbols) or occupied[start] or occupied[stop]:
            continue
        if (symbols[start], symbols[stop]) != pair:
            continue
        occupied[start] = True
        occupied[stop] = True
        spans.append((start, stop))
    cells = []
    cursor = 0
    while cursor < len(symbols):
        span = next((item for item in spans if item[0] == cursor), None)
        if span is None:
            cells.append((cursor,))
            cursor += 1
        else:
            cells.append(span)
            cursor = span[1] + 1
    return cells


class MergeControlTests(unittest.TestCase):
    def test_count_and_ratio_select_the_same_pair(self):
        grid = [
            [1, 2, 1, 2],
            [1, 2, 3, 4],
            [3, 4, 3, 4],
            [9, 9, 9, 8],
        ]
        by_count = GridMerger(grid, "count")
        by_ratio = GridMerger(grid, "ratio")
        self.assertEqual(by_count.chosen, by_ratio.chosen)
        self.assertEqual(len(by_count.tokens), len(by_ratio.tokens))
        self.assertEqual(by_count.values(), grid)
        self.assertEqual(by_ratio.values(), grid)

    def test_published_spatial_term_changes_only_a_close_frequency(self):
        close_counts = (2, 2)
        wide_counts = (5, 1)
        close_total = sum(close_counts)
        wide_total = sum(wide_counts)
        close_priority = (
            _priority(close_counts[0], close_total, [(0, 1), (1, 0)]),
            _priority(close_counts[1], close_total, [(0, 1), (0, 1)]),
        )
        wide_priority = (
            _priority(wide_counts[0], wide_total, [(0, 1), (1, 0), (0, 1), (1, 0), (0, -1)]),
            _priority(wide_counts[1], wide_total, [(0, 1)]),
        )
        self.assertNotEqual(_first_index(close_counts), _first_index(close_priority))
        self.assertEqual(_first_index(wide_counts), _first_index(wide_priority))

        tied = [
            [1, 2, 3, 3],
            [4, 5, 6, 7],
            [1, 8, 9, 10],
            [2, 3, 3, 11],
        ]
        by_count = GridMerger(tied, "count")
        by_priority = GridMerger(tied, "priority")
        self.assertEqual(by_count.chosen[0], (1, 2))
        self.assertEqual(by_priority.chosen[0], (2, 3))
        self.assertEqual(len(by_count.tokens), 12)
        self.assertEqual(len(by_priority.tokens), 10)
        self.assertEqual(by_count.values(), tied)
        self.assertEqual(by_priority.values(), tied)

    def test_unknown_maximum_changes_length_and_keeps_values(self):
        tokenizer = _tokenizer()
        tokenizer.vocab.add(("a", "b"), "M", (1, 2, 2))
        tokenizer.freq_counter.total_count = 4
        tokenizer.freq_counter.count_table = {("c", "d"): 3, ("a", "b"): 1}
        state = State(torch.zeros(1, 1, 2, 2, dtype=torch.uint8))
        state.joined = ["c", "d", "a", "b"]
        tokenizer.merge_pair(0, state, (1, 2, 2), update=False)
        self.assertEqual(state.joined, ["c", "d", "a", "b"])

        skipped = State(torch.zeros(1, dtype=torch.uint8)).merge(["c", "d", "a", "b"], ("a", "b"), "M")
        self.assertEqual(skipped, ["c", "d", "M"])
        self.assertNotEqual(len(state.joined), len(skipped))
        self.assertEqual(skipped[2], "M")

    def test_rectangle_chain_and_two_cell_constellation_lengths(self):
        grid = [
            [1, 3, 2, 4],
            [7, 11, 8, 12],
            [1, 5, 2, 6],
            [9, 13, 10, 14],
        ]
        order = _patch_order(4, 4)
        ordered = [grid[index // 4][index % 4] for index in order]
        rectangle, _chosen, _symbols = _merge_sequence(ordered, "count")
        constellation = GridMerger(grid, "count")
        self.assertEqual(ordered[:4], [1, 2, 1, 2])
        once = State(torch.zeros(1, dtype=torch.uint8)).merge(ordered, (1, 2), "M")
        self.assertEqual(len(once), 14)
        self.assertEqual(len(rectangle), 14)
        self.assertEqual(len(constellation.tokens), 16)
        self.assertEqual(constellation.values(), grid)
        covered = [cell for token in rectangle for cell in token["cells"]]
        self.assertEqual(sorted(covered), list(range(16)))

        stripes = torch.tensor(
            [[[[1, 2, 1, 2], [1, 2, 1, 2], [3, 4, 3, 4], [3, 4, 3, 4]]]],
            dtype=torch.uint8,
        )
        tokenizer = _tokenizer()
        tokenizer.train(TensorDataset(stripes), 0)
        encoded = tokenizer.encode(stripes)
        self.assertTrue(torch.equal(tokenizer.decode(encoded), stripes))
        self.assertEqual(len(encoded.sequences[0]), 1)

    def test_overlap_ban_matches_left_merge_only_in_the_same_order(self):
        symbols = ["a", "a", "a"]
        pair = ("a", "a")
        _merged, left_cells = _partition_from_merge(symbols, pair, "M")
        forward = _partition_from_ban(symbols, pair, [0, 1])
        reverse = _partition_from_ban(symbols, pair, [1, 0])
        self.assertEqual(left_cells, forward)
        self.assertNotEqual(left_cells, reverse)
        self.assertEqual(len(left_cells), len(reverse))
        self.assertEqual(sorted(index for span in left_cells for index in span), [0, 1, 2])
        self.assertEqual(sorted(index for span in reverse for index in span), [0, 1, 2])

    def test_symbol_limit_stops_before_the_next_merge(self):
        grid = [[1, 1], [1, 1]]
        open_ended = GridMerger(grid, "count")
        capped = GridMerger(grid, "count", new_symbol_limit=0)
        self.assertLess(len(open_ended.tokens), 4)
        self.assertEqual(len(capped.tokens), 4)
        self.assertEqual(capped.values(), grid)
        self.assertIsNone(open_ended.trace)

    def test_first_fork_is_the_known_close_pair(self):
        tied = [
            [1, 2, 3, 3],
            [4, 5, 6, 7],
            [1, 8, 9, 10],
            [2, 3, 3, 11],
        ]
        result = first_fork(tied)
        self.assertEqual(result["fork"]["step"], 0)
        self.assertEqual(result["fork"]["token_count"], 16)
        self.assertEqual(result["fork"]["count_pair"], (1, 2))
        self.assertEqual(result["fork"]["spatial_pair"], (2, 3))
        count_note = result["fork"]["count_annotation"]
        spatial_note = result["fork"]["spatial_annotation"]
        self.assertEqual(count_note["members"], ["原始", "原始"])
        self.assertEqual(spatial_note["members"], ["原始", "原始"])
        self.assertIn(count_note["direction"], ("水平", "垂直", "深层"))
        self.assertIn(spatial_note["direction"], ("水平", "垂直", "深层"))
        self.assertGreaterEqual(count_note["count_score"], spatial_note["count_score"])
        self.assertGreaterEqual(spatial_note["spatial_score"], count_note["spatial_score"])
        self.assertEqual(result["count_final_tokens"], 12)
        self.assertEqual(result["spatial_final_tokens"], 10)
        self.assertTrue(result["count_lossless"])
        self.assertTrue(result["spatial_lossless"])
