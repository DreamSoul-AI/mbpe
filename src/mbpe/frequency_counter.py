import torch
import numpy as np
from collections import defaultdict
from .utils import update_vocab


class FrequencyCounter:
    def __init__(self, min_freq):
        self.min_freq = min_freq
        self.reset()

    def reset(self):
        self.count_table = {}
        self.total_count = 0
        return

    def get_freq(self, symbols):
        freqs = []
        for symbol in symbols:
            count = self.count_table.get(symbol, 0)
            freq = count / self.total_count
            freqs.append(freq)
        return freqs

    def update(self, symbols):
        symbol_tensors = torch.tensor(symbols)
        unique_symbols, counts = torch.unique(symbol_tensors, sorted=False, return_counts=True, dim=0)
        self.total_count += len(symbols)
        for symbol, count in zip(symbols, counts):
            if symbol not in self.count_table:
                if count / len(symbols) > self.min_freq['freq_counter']:
                    self.count_table[symbol] = count
            else:
                self.count_table[symbol] += count
        return

    def filter_symbols(self, symbols):
        in_symbols = []
        in_indices = []
        out_symbols = []
        out_indices = []
        for idx, symbol in enumerate(symbols):
            if symbol in self.count_table:
                in_symbols.append(symbol)
                in_indices.append(idx)
            else:
                out_symbols.append(symbol)
                out_indices.append(idx)
        return in_symbols, in_indices, out_symbols, out_indices







        # self.min_entrance_freq = min_entrance_freq
        # self.min_root_freq = min_root_freq  # Minimum frequency threshold for root vocabulary
        # self.global_freq_table = defaultdict(lambda: {'global_count': 0, 'global_frequency': 0.0})
        # self.max_merge_pair = {'pair': None, 'global_frequency': 0.0}
        # self.total_count = 0

    # def get_global_freq_table(self):
    #     return self.global_freq_table
    #
    # def get_max_merge_pair(self):
    #     return self.max_merge_pair

    # def initialize_local_freq_table(self):
    #     """Step 1: Initialize local_freq_table."""
    #     return defaultdict(lambda: {'count': 0, 'frequency': 0.0})


    def count_frequencies_in_tensor(self, tensor, local_freq_table):  # TODO: what does item mean here? seems werid
        """Step 2: Count frequency of each item in the current tensor."""
        output, counts = torch.unique(tensor, sorted=False, return_counts=True, dim=0)
        output_list = output.flatten(start_dim=1).tolist()
        for i in range(len(output_list)):
            local_freq_table[tuple(output_list[i])]['count'] += counts[i].item()
        return local_freq_table

    def count_frequencies_in_list(self, joined_list, local_freq_table):  # TODO: should avoid this
        """Step 2: Count frequency of each item in the current list."""
        for i in range(len(joined_list) - 1):
            pair = (joined_list[i], joined_list[i + 1])
            local_freq_table[pair]['count'] += 1
        return local_freq_table

    def calculate_local_frequencies(self, local_freq_table, size):  # TODO: should store in tensors to avoid for loop
        """Step 3: Calculate frequency for each item in the batch."""
        for item in local_freq_table:
            local_freq_table[item]['frequency'] = local_freq_table[item]['count'] / size
        return local_freq_table

    def update_global_freq_table(self, local_freq_table, size,
                                 update_merge_pair=False):  # TODO: why have separate local and global tables
        """Step 4: Update global_freq_table based on the local_freq_table."""
        for item, info in local_freq_table.items():
            frequency = info['frequency']

            # Only process items whose frequency is greater than or equal to min_entrance_freq
            if item in self.global_freq_table:
                self.global_freq_table[item]['global_count'] += info['count']
                self.global_freq_table[item]['global_frequency'] = (
                        self.global_freq_table[item]['global_count'] / (self.total_count + size)
                )
            else:
                if frequency >= self.min_entrance_freq:
                    self.global_freq_table[item] = {
                        'global_count': info['count'],
                        'global_frequency': info['count'] / (self.total_count + size)
                    }

            if update_merge_pair and self.max_merge_pair['global_frequency'] < self.global_freq_table[item][
                'global_frequency']:
                self.max_merge_pair['pair'] = item
                self.max_merge_pair['global_frequency'] = self.global_freq_table[item]['global_frequency']

    def update_root_vocabulary(self, vocab, inverse_vocab):  # TODO: duplicate with process root
        """Update root_vocabulary by including tuples with frequency >= min_root_freq."""
        for item, info in self.global_freq_table.items():
            if info['global_frequency'] >= self.min_root_freq and item not in inverse_vocab:
                update_vocab(vocab, inverse_vocab, item, str(len(vocab)))

    def update_freq_tables(self, tensor, vocab, inverse_vocab):
        """Update both local and global frequency tables."""
        # Initialize the local frequency table for the current batch
        # local_freq_table = self.initialize_local_freq_table()

        # Count frequencies in the current batch
        local_freq_table = self.count_frequencies_in_tensor(tensor, local_freq_table)

        # Calculate frequency for each item in the current tensor
        local_freq_table = self.calculate_local_frequencies(local_freq_table, len(tensor))

        # Update the global frequency table based on the local frequency table
        self.update_global_freq_table(local_freq_table, len(tensor))

        # Update the total_count to reflect the total number of processed items
        self.total_count += len(tensor)

        # After updating the global frequency table, update root_vocabulary
        self.update_root_vocabulary(vocab, inverse_vocab)

        return local_freq_table

    def update_merge_freq_tables(self, mixed_list):
        """Update merge frequency tables."""
        if len(self.global_freq_table) > 0:
            self.global_freq_table = defaultdict(lambda: {'global_count': 0, 'global_frequency': 0.0})
        local_freq_table = self.initialize_local_freq_table()
        local_freq_table = self.count_frequencies_in_list(mixed_list, local_freq_table)
        local_freq_table = self.calculate_local_frequencies(local_freq_table, len(mixed_list))
        self.update_global_freq_table(local_freq_table, len(mixed_list), update_merge_pair=True)
        self.total_count += len(mixed_list)
        return local_freq_table

