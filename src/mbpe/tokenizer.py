import threading
from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor

from .utils import *
from .patch import *
from .state import State
from .frequency_counter import FrequencyCounter
from .vocab import Vocab


class Tokenizer:
    def __init__(self, min_freq, max_codeword_size, batch_size=1, max_workers=None):
        super().__init__()
        self.vocab = Vocab()
        self.freq_counter = FrequencyCounter(min_freq)
        self.codeword_sizes = self.find_tuple_shapes(max_codeword_size)
        print('Found codeword Sizes:{}'.format(self.codeword_sizes))
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
    def train(self, dataset, data_name):
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        states = {}
        for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training"):
            data = data[data_name]
            for i in range(len(data)):  # TODO: add multithread/multiprocess here (later)
                data_i = data[[i]]  # Add first dimension
                index_tracker = batch_idx * self.batch_size + i
                if index_tracker not in states:
                    states[index_tracker] = State(data_i)
                for m, codeword_size in enumerate(self.codeword_sizes):
                    patchify = Patchify(codeword_size)
                    states[index_tracker] = self.patchify(m, states[index_tracker], patchify)
                    states[index_tracker] = self.make_root(m, states[index_tracker], codeword_size)
                    states[index_tracker] = self.merge_pair(m, states[index_tracker], codeword_size)
                    # print(states[index_tracker])
        return

    def patchify(self, step, state, patchify):
        if step > 0:
            scale_factor = self.get_scale_factor(state.size, patch_size=self.codeword_sizes[step])
            # TODO: need to add scale_factor below one case, increment for every scale_factor indices
            symbols, symbol_indices, codewords, codeword_indices = state.split(state.joined, scale_factor)
            # print(scale_factor)
            # print(len(symbols), len(symbol_indices), len(codewords), len(codeword_indices))
            # print(symbol_indices)
            # print(codeword_indices)
            data = tuple_to_tensor(symbols, state.size, state.dtype, is_batch=False)
            state.update(data=data, symbols=symbols, symbol_indices=symbol_indices,
                         codewords=codewords, codeword_indices=codeword_indices,
                         codeword_size=self.codeword_sizes[step])

        codeword_size = patchify.patch_size
        data = patchify(state.data)
        symbols = tensor_to_tuple(data, codeword_size, is_batch=False)
        state.update(data=data, symbols=symbols)
        # print(state)
        return state

    def make_root(self, step, state, codeword_size, update=True):
        ignore_threshold = step == len(self.codeword_sizes) - 1
        if update:
            self.freq_counter.update(state.symbols, ignore_threshold)
            threshold = None if ignore_threshold else self.freq_counter.min_freq['root']
            root_symbols, root_indices, non_root_symbols, non_root_indices = \
                self.freq_counter.filter_symbols(state.symbols, threshold)
            root_codewords = self.vocab.update(root_symbols, codeword_size)
        else:
            root_symbols, root_indices, non_root_symbols, non_root_indices = \
                self.vocab.filter_symbols(state.symbols, codeword_size)
            root_codewords = self.vocab.get_codewords(root_symbols, codeword_size)

        data = state.data[non_root_indices]

        if step > 0:
            root_indices = torch.tensor(state.symbol_indices)[root_indices].tolist()
            non_root_indices = torch.tensor(state.symbol_indices)[non_root_indices].tolist()

        symbols = non_root_symbols
        symbol_indices = non_root_indices
        state.codewords.extend(root_codewords)
        state.codeword_indices[codeword_size].extend(root_indices)

        joined = state.join(symbols, symbol_indices, state.codewords, state.codeword_indices[codeword_size])
        state.update(data=data, symbols=symbols, symbol_indices=symbol_indices, joined=joined,
                     codeword_size=codeword_size)
        # print(state)
        return state

    def merge_pair(self, step, state, codeword_size, update=True):
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
                codeword = self.vocab.update(max_pair, codeword_size)
            else:
                codeword = self.vocab.get_codeword(max_pair, codeword_size)
            if codeword is None:
                break
            joined = state.merge(state.joined, max_pair, codeword)
            state.update(joined=joined)

        if step == len(self.codeword_sizes) - 1:
            state.finalize()
        return state

    @torch.no_grad()
    def encode(self, data):
        is_batched = data.dim() > 1 and data.size(0) > 1

        # Single batched sample
        if not is_batched:
            state = State(data)
            for m, codeword_size in enumerate(self.codeword_sizes):
                patchify = Patchify(codeword_size)
                state = self.patchify(m, state, patchify)
                state = self.make_root(m, state, codeword_size, update=False)
                state = self.merge_pair(m, state, codeword_size, update=False)
            return state.joined

        # Batched samples
        result = []
        for i in range(len(data)):
            data_i = data[[i]]
            state = State(data_i)
            for m, codeword_size in enumerate(self.codeword_sizes):
                patchify = Patchify(codeword_size)
                state = self.patchify(m, state, patchify)
                state = self.make_root(m, state, codeword_size, update=False)
                state = self.merge_pair(m, state, codeword_size, update=False)
            result.append(state.joined)

        return result

    @torch.no_grad()  # TODO: refactor, need to revise and test
    def decode(self, state, data_shape, data_dtype):
        codeword_sizes = list(reversed(self.codeword_sizes))
        current_state = state

        for m, codeword_size in enumerate(codeword_sizes):
            print(m, codeword_size)
            codeword_size = tuple(codeword_size)
            is_final = (m == len(codeword_sizes) - 1)

            decoded = []
            if is_final:
                accumulate = None
            for i in range(len(current_state.joined)):
                codeword_i = current_state.joined[i]
                get_symbol = partial(self.vocab.get_symbol, codeword_size=codeword_size)
                decoded_i = dfs(codeword_i, get_symbol)
                if is_final:
                    tuple_length = math.prod(codeword_size)
                    for tup in decoded_i:
                        if len(tup) != tuple_length:
                            if accumulate is None:
                                accumulate = tup
                            else:
                                accumulate += tup
                            if len(accumulate) == tuple_length:
                                decoded.extend([accumulate])
                                accumulate = None
                        else:
                            decoded.extend([tup])
                else:
                    decoded.extend(decoded_i)
            print(f"Decoded: {decoded}")

            current_state.joined = decoded

        data = tuple_to_tensor(current_state.joined, [len(current_state.joined)] + list(codeword_size), data_dtype, is_batch=False)
        # print(data)
        print(f"Final data size: {data.size()}")

        reconstruct = Reconstruct(data_shape[-len(codeword_size):])
        reconstructed = reconstruct(data)
        # print(reconstructed)
        # print(reconstructed.size())

        return reconstructed

    def get_scale_factor(self, size, patch_size=None, dim_index=None):
        patch_size = patch_size if patch_size is not None else patch_size
        if dim_index is None:
            dim_index = list(range(len(size) - len(patch_size), len(size)))
        for i, dim in enumerate(dim_index):
            factor = size[dim] // patch_size[i]
            if factor != 1:
                return factor
        return 1
