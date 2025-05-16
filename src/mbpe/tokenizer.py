import threading
import torch
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from .utils import *
from .patch import *
from .state import State
from .frequency_counter import FrequencyCounter
from .vocab import Vocab


class Tokenizer:
    def __init__(self, min_freq, batch_size=1, max_workers=None):
        super().__init__()
        self.vocab = Vocab()
        self.freq_counter = FrequencyCounter(min_freq)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._lock = threading.Lock()

    def find_tuple_shapes(self, dim):
        """
        Find all possible shapes of tuples based on the given dimensions.

        Args:
            dim (tuple): A tuple of dimensions.

        Returns:
            shapes: A list of tuples representing all possible shapes.
        """

        # Find divisors for each dimension in desceding order
        # Each smaller divisor must also be divisible by the previous larger divisor
        # e.g. (2, 2) -> (2, 2), (1, 2), (1, 1); (6, 2) -> (6, 2), (3, 2), (1, 2), (1, 1)
        dim = list(dim)
        divisors = []
        for d in dim:
            tmp = []
            new_divisor = d
            for i in range(d - 1, 0, -1):
                if d % i == 0 and new_divisor % i == 0:
                    tmp.append(i)
                    new_divisor = i
            divisors.append(tmp)

        # Generate all possible shapes
        shapes = [dim]
        current_dim = dim
        for i, div in enumerate(divisors):
            for d in div:
                new_shape = current_dim.copy()
                new_shape[i] = d
                current_dim = new_shape
                shapes.append(new_shape)

        return shapes


    @torch.no_grad()
    def train(self, dataset, data_name, max_codeword_size, dim_index):
        codeword_sizes = self.find_tuple_shapes(max_codeword_size)
        print(codeword_sizes)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        print('Found codeword Sizes:{}'.format(codeword_sizes))
        states = {}
        for m, codeword_size in enumerate(codeword_sizes):
            patchify = Patchify(codeword_size, dim_index)
            is_last = m == len(codeword_sizes) - 1
            print(codeword_size, is_last)
            index_tracker = 0
            for batch_idx, data in enumerate(data_loader):
                data = data[data_name]
                for i in range(len(data)):  # TODO: add multithread/multiprocess here (later)
                    print(i)
                    data_i = data[i]
                    if index_tracker not in states:
                        states[index_tracker] = State(data_i)
                    states[index_tracker] = self.make_root(states[index_tracker], patchify, is_last)
                    states[index_tracker] = self.merge_pair(states[index_tracker])
                    print(states[index_tracker])
                    print(self.vocab)
                    print(self.freq_counter)
                    index_tracker += 1
        return

    def make_root(self, state, patchify, is_last):
        codeword_size = patchify.patch_size
        unshuffled_data = patchify(state.data, is_batch=False)
        symbols = tensor_to_tuple(unshuffled_data, codeword_size, is_batch=False)
        self.freq_counter.update(symbols, is_last)
        threshold = None if is_last else self.freq_counter.min_freq['root']
        root_symbols, root_symbol_indices, non_root_symbols, non_root_symbol_indices = \
            self.freq_counter.filter_symbols(symbols, threshold)
        codewords = self.vocab.update(root_symbols)
        non_root_data = unshuffled_data[non_root_symbol_indices]
        joined = state.join(non_root_symbols, non_root_symbol_indices, codewords, root_symbol_indices)
        state.update(data=non_root_data, symbols=non_root_symbols, symbol_indices=non_root_symbol_indices,
                     codewords=codewords, codeword_indices=root_symbol_indices, joined=joined)
        return state

    def merge_pair(self, state):
        while True:
            pairs = state.merge(state.joined)
            self.freq_counter.update(pairs)
            freqs = self.freq_counter.get_freqs(pairs)
            max_idx = np.argmax(freqs)
            max_pair, max_freq = pairs[max_idx], freqs[max_idx]
            if max_freq < self.freq_counter.min_freq['merge']:
                break
            codeword = self.vocab.update(max_pair)
            joined = state.merge_symbol(state.joined, max_pair, codeword)

            symbols = []
            symbol_indices = []
            codewords = []
            codeword_indices = []
            for i, item in enumerate(joined):
                if isinstance(item, tuple):
                    symbols.append(item)
                    symbol_indices.append(i)
                else:
                    codewords.append(item)
                    codeword_indices.append(i)
            data = tuple_to_tensor(symbols, state.size, state.dtype, is_batch=False)
            state.update(data=data, symbols=symbols, symbol_indices=symbol_indices,
                         codewords=codewords, codeword_indices=codeword_indices, joined=joined)
        return state
