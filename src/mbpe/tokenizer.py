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
        shapes = [tuple(dim)]
        current_dim = dim
        for i, div in enumerate(divisors):
            for d in div:
                new_shape = current_dim.copy()
                new_shape[i] = d
                current_dim = new_shape
                shapes.append(tuple(new_shape))

        return shapes

    @torch.no_grad()
    def train(self, dataset, data_name, max_codeword_size, dim_index):
        codeword_sizes = self.find_tuple_shapes(max_codeword_size)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        print('Found codeword Sizes:{}'.format(codeword_sizes))
        states = {}
        for m, codeword_size in enumerate(codeword_sizes):
            patchify = Patchify(codeword_size, dim_index)
            is_first = m == 0
            is_last = m == len(codeword_sizes) - 1
            # print(m, codeword_size, is_last)
            index_tracker = 0
            for batch_idx, data in enumerate(data_loader):
                data = data[data_name]
                for i in range(len(data)):  # TODO: add multithread/multiprocess here (later)
                    data_i = data[i]
                    if index_tracker not in states:
                        states[index_tracker] = State(data_i)
                    states[index_tracker] = self.patchify(states[index_tracker], patchify, is_first)
                    states[index_tracker] = self.make_root(states[index_tracker], codeword_size, is_first, is_last)
                    states[index_tracker] = self.merge_pair(states[index_tracker], codeword_size, is_last)
                    index_tracker += 1
            # print(states[index_tracker - 1])
        return

    def patchify(self, state, patchify, is_first):
        if not is_first:
            scale_factor = patchify.get_scale_factor(state.size)
            symbols, symbol_indices, codewords, codeword_indices = state.split(state.joined, scale_factor)
            data = tuple_to_tensor(symbols, state.size, state.dtype, is_batch=False)
            state.update(data=data, symbols=symbols, symbol_indices=symbol_indices,
                         codewords=codewords, codeword_indices=codeword_indices)

        codeword_size = patchify.patch_size
        data = patchify(state.data, is_batch=False)
        symbols = tensor_to_tuple(data, codeword_size, is_batch=False)
        state.update(data=data, symbols=symbols)
        return state

    def make_root(self, state, codeword_size, is_first, is_last, update=True):
        if update:
            self.freq_counter.update(state.symbols, is_last)
            threshold = None if is_last else self.freq_counter.min_freq['root']
            root_symbols, root_indices, non_root_symbols, non_root_indices = \
                self.freq_counter.filter_symbols(state.symbols, threshold)
            root_codewords = self.vocab.update(root_symbols)
        else:
            root_symbols, root_indices, non_root_symbols, non_root_indices = \
                self.vocab.filter_symbols(state.symbols, codeword_size)
            root_codewords = self.vocab.get_codewords(root_symbols)

        data = state.data[non_root_indices]

        if not is_first:
            root_indices = torch.tensor(state.symbol_indices)[root_indices].tolist()
            non_root_indices = torch.tensor(state.symbol_indices)[non_root_indices].tolist()

        symbols = non_root_symbols
        symbol_indices = non_root_indices
        state.codewords.extend(root_codewords)
        state.codeword_indices.extend(root_indices)

        joined = state.join(symbols, symbol_indices, state.codewords, state.codeword_indices)
        state.update(data=data, symbols=symbols, symbol_indices=symbol_indices, joined=joined)
        # print(state)
        return state

    def merge_pair(self, state, codeword_size, is_last, update=True):
        while True:
            pairs = state.pair(state.joined)
            if update:
                self.freq_counter.update(pairs)
            freqs = self.freq_counter.get_freqs(pairs)
            max_idx = np.argmax(freqs)
            max_pair, max_freq = pairs[max_idx], freqs[max_idx]
            if update:
                if max_freq < self.freq_counter.min_freq['merge']:
                    break
                codeword = self.vocab.update(max_pair)
            else:
                codeword = self.vocab.get_codeword(max_pair, codeword_size)
            if codeword is None:
                break
            # print(max_pair, max_freq, codeword)
            joined = state.merge(state.joined, max_pair, codeword)
            state.update(joined=joined)
        if is_last:
            state.finalize()
        return state

    @torch.no_grad()
    def encode(self, data, max_codeword_size, dim_index):
        codeword_sizes = self.find_tuple_shapes(max_codeword_size)
        state = State(data)
        for m, codeword_size in enumerate(codeword_sizes):
            # print(m, codeword_size)
            codeword_size = tuple(codeword_size)
            patchify = Patchify(codeword_size, dim_index)
            is_first = m == 0
            is_last = m == len(codeword_sizes) - 1
            state = self.patchify(state, patchify, is_first)
            state = self.make_root(state, codeword_size, is_first, is_last, update=False)
            state = self.merge_pair(state, codeword_size, is_last, update=False)
            # print(state)
        return state

    @torch.no_grad()
    def decode(self, state, data_shape, data_dtype):
        joined = state.joined
        decoded = []
        for i in range(len(joined)):
            codeword_i = joined[i]
            decoded_i = dfs(codeword_i, self.vocab.get_symbol)
            decoded.extend(decoded_i)
        decoded = torch.tensor(decoded, dtype=data_dtype).view(*data_shape)
        return decoded
