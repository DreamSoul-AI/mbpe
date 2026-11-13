import torch
import numpy as np
from collections import Counter


class FrequencyCounter:
    def __init__(self, min_freq):
        self.min_freq = min_freq
        self.reset()

    def reset(self):
        self.count_table = {}
        self.total_count = 0
        return

    def get_freq(self, symbol):
        count = self.count_table.get(symbol, 0)
        freq = count / self.total_count
        return freq

    def get_freqs(self, symbols):
        freqs = []
        for symbol in symbols:
            freq = self.get_freq(symbol)
            freqs.append(freq)
        return freqs

    def update(self, symbols, ignore_threshold=False):
        counter = Counter(symbols)
        self.total_count += len(symbols)
        for symbol, count in counter.items():
            if symbol not in self.count_table:
                if count / len(symbols) >= self.min_freq['freq_counter'] or ignore_threshold:
                    self.count_table[symbol] = count
            else:
                self.count_table[symbol] += count
        return

    def filter_symbols(self, symbols, threshold=None):
        in_symbols = []
        in_indices = []
        out_symbols = []
        out_indices = []
        for idx, symbol in enumerate(symbols):
            if threshold is None:
                if symbol in self.count_table:
                    in_symbols.append(symbol)
                    in_indices.append(idx)
                else:
                    out_symbols.append(symbol)
                    out_indices.append(idx)
            else:
                if self.get_freq(symbol) >= threshold:
                    in_symbols.append(symbol)
                    in_indices.append(idx)
                else:
                    out_symbols.append(symbol)
                    out_indices.append(idx)
        return in_symbols, in_indices, out_symbols, out_indices

    def __repr__(self):
        num_entries = len(self.count_table)
        preview_limit = 20
        sorted_items = sorted(self.count_table.items(), key=lambda x: -x[1])
        preview = sorted_items[:preview_limit]
        preview_str = ", ".join(f"{repr(k)}: {v}" for k, v in preview)
        if num_entries > preview_limit:
            preview_str += ", ..."
        return (f"FrequencyCounter(min_freq={self.min_freq}, total_count={self.total_count}, "
                f"{num_entries} symbols: {{{preview_str}}})")
