import math
import threading
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TensorEncoding:
    """One encode result. Token order is the reconstruction order."""

    sequences: tuple
    shape: tuple
    dtype: torch.dtype

    def __post_init__(self):
        if type(self.sequences) is not tuple or any(type(sequence) is not tuple for sequence in self.sequences):
            raise TypeError("sequences must be a tuple of token tuples")
        if type(self.shape) is not tuple or not self.shape or any(type(dim) is not int or dim < 1 for dim in self.shape):
            raise ValueError("shape must be a tuple of positive integers")
        if len(self.sequences) != self.shape[0]:
            raise ValueError("sequence count does not match the leading batch size")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch dtype")


class Tokenizer:
    def __init__(self, min_freq, max_codeword_size, batch_size=1, max_workers=None,
                 max_symbols=None, seed=None):
        super().__init__()
        self.vocab = Vocab()
        self.freq_counter = FrequencyCounter(min_freq)
        self.codeword_sizes = self.find_tuple_shapes(max_codeword_size)
        print('Found codeword Sizes:{}'.format(self.codeword_sizes))
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_symbols = max_symbols
        self.seed = seed
        self.merge_order = []
        self._lock = threading.Lock()

    def _symbol_cap_reached(self):
        return self.max_symbols is not None and len(self.vocab) >= self.max_symbols

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
                self._validate_input(data_i)
                index_tracker = batch_idx * self.batch_size + i
                if index_tracker not in states:
                    states[index_tracker] = State(data_i)
                for m, codeword_size in enumerate(self.codeword_sizes):
                    if self._symbol_cap_reached():
                        return
                    patchify = Patchify(codeword_size)
                    states[index_tracker] = self.patchify(m, states[index_tracker], patchify)
                    states[index_tracker] = self.make_root(m, states[index_tracker], codeword_size)
                    states[index_tracker] = self.merge_pair(m, states[index_tracker], codeword_size)
                    if self._symbol_cap_reached():
                        return
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
            if self.max_symbols is not None:
                root_symbols, root_indices, non_root_symbols, non_root_indices = self._limit_new_symbols(
                    root_symbols, root_indices, non_root_symbols, non_root_indices)
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

    def _limit_new_symbols(self, root_symbols, root_indices, non_root_symbols, non_root_indices):
        kept_symbols, kept_indices = [], []
        held_symbols, held_indices = [], []
        for symbol, index in zip(root_symbols, root_indices):
            reserved = sum(item not in self.vocab.symbol2codeword for item in kept_symbols)
            if symbol not in self.vocab.symbol2codeword and len(self.vocab) + reserved >= self.max_symbols:
                held_symbols.append(symbol)
                held_indices.append(index)
            else:
                kept_symbols.append(symbol)
                kept_indices.append(index)
        return (kept_symbols, kept_indices,
                held_symbols + list(non_root_symbols), held_indices + list(non_root_indices))

    def merge_pair(self, step, state, codeword_size, update=True):
        while True:
            if update and self._symbol_cap_reached():
                break
            pairs = state.pair(state.joined)
            if not pairs:
                break
            if update:
                self.freq_counter.update(pairs)
            elif self.freq_counter.total_count == 0:
                break
            freqs = self.freq_counter.get_freqs(pairs)
            max_idx = np.argmax(freqs)
            max_pair, max_freq = pairs[max_idx], freqs[max_idx]
            if update:
                if max_freq < self.freq_counter.min_freq['merge']:
                    break
                seen = max_pair in self.vocab.symbol2codeword
                codeword = self.vocab.update(max_pair, codeword_size)
                if not seen:
                    self.merge_order.append((max_pair, codeword, tuple(codeword_size)))
            else:
                codeword = self.vocab.get_codeword(max_pair, codeword_size)
            if codeword is None:
                break
            joined = state.merge(state.joined, max_pair, codeword)
            state.update(joined=joined)

        if step == len(self.codeword_sizes) - 1:
            state.finalize()
        return state

    def _validate_input(self, data):
        if not isinstance(data, torch.Tensor):
            raise TypeError("Tokenizer expects a torch.Tensor")
        if data.device.type != "cpu":
            raise ValueError("Tokenizer expects a CPU tensor")
        if data.dtype.is_floating_point or data.dtype.is_complex or data.dtype == torch.bool:
            raise ValueError("Tokenizer expects an integer tensor")
        if data.ndim < 1 or int(data.shape[0]) < 1 or data.numel() == 0:
            raise ValueError("Tokenizer expects a non-empty leading batch")
        codeword = self.codeword_sizes[0]
        if data.ndim <= len(codeword):
            raise ValueError("tensor rank must be greater than the codeword rank")
        spatial = tuple(int(size) for size in data.shape[-len(codeword):])
        if any(size % step != 0 for size, step in zip(spatial, codeword)):
            raise ValueError(f"spatial shape {spatial} is not divisible by codeword shape {tuple(codeword)}")

    @torch.no_grad()
    def encode(self, data):
        self._validate_input(data)
        vocab_size = len(self.vocab)
        frequency_total = self.freq_counter.total_count
        sequences = []
        for index in range(int(data.shape[0])):
            state = State(data[index:index + 1])
            for step, codeword_size in enumerate(self.codeword_sizes):
                state = self.patchify(step, state, Patchify(codeword_size))
                state = self.make_root(step, state, codeword_size, update=False)
                state = self.merge_pair(step, state, codeword_size, update=False)
            sequences.append(tuple(state.joined))
        if len(self.vocab) != vocab_size or self.freq_counter.total_count != frequency_total:
            raise RuntimeError("encode must not modify the vocabulary or frequency counts")
        return TensorEncoding(tuple(sequences), tuple(int(dim) for dim in data.shape), data.dtype)

    def _decode_sequence(self, tokens, data_shape, data_dtype):
        codeword_sizes = [tuple(size) for size in reversed(self.codeword_sizes)]
        current = list(tokens)
        for m, codeword_size in enumerate(codeword_sizes):
            is_final = m == len(codeword_sizes) - 1
            decoded = []
            accumulate = None
            for codeword_i in current:
                get_symbol = partial(self.vocab.get_symbol, codeword_size=codeword_size)
                decoded_i = dfs(codeword_i, get_symbol)
                if not is_final:
                    decoded.extend(decoded_i)
                    continue
                tuple_length = math.prod(codeword_size)
                for tup in decoded_i:
                    if not isinstance(tup, tuple):
                        raise ValueError(f"unresolved codeword: {tup!r}")
                    if len(tup) != tuple_length:
                        accumulate = tup if accumulate is None else accumulate + tup
                        if len(accumulate) == tuple_length:
                            decoded.append(accumulate)
                            accumulate = None
                        elif len(accumulate) > tuple_length:
                            raise ValueError("encoded patches do not align with the codeword shape")
                    elif accumulate is not None:
                        raise ValueError("encoded patches do not align with the codeword shape")
                    else:
                        decoded.append(tup)
            if accumulate is not None:
                raise ValueError("encoded sequence does not complete a patch")
            current = decoded
        if not current:
            raise ValueError("encoded sequence is empty")
        codeword_size = codeword_sizes[-1]
        data = tuple_to_tensor(current, [len(current)] + list(codeword_size), data_dtype, is_batch=False)
        reconstructed = Reconstruct(data_shape[-len(codeword_size):])(data)
        if tuple(reconstructed.shape) != tuple(data_shape) or reconstructed.dtype != data_dtype:
            raise ValueError(
                f"reconstruction produced {tuple(reconstructed.shape)} {reconstructed.dtype}, "
                f"expected {tuple(data_shape)} {data_dtype}"
            )
        return reconstructed

    @torch.no_grad()
    def decode(self, encoding):
        if not isinstance(encoding, TensorEncoding):
            raise TypeError("decode expects the TensorEncoding returned by encode")
        vocab_size = len(self.vocab)
        frequency_total = self.freq_counter.total_count
        sample_shape = (1,) + encoding.shape[1:]
        restored = torch.cat(
            [self._decode_sequence(sequence, sample_shape, encoding.dtype) for sequence in encoding.sequences],
            dim=0,
        )
        if len(self.vocab) != vocab_size or self.freq_counter.total_count != frequency_total:
            raise RuntimeError("decode must not modify the vocabulary or frequency counts")
        if tuple(restored.shape) != encoding.shape or restored.dtype != encoding.dtype:
            raise ValueError(
                f"decoded batch is {tuple(restored.shape)} {restored.dtype}, "
                f"expected {encoding.shape} {encoding.dtype}"
            )
        return restored

    def get_scale_factor(self, size, patch_size=None, dim_index=None):
        patch_size = patch_size if patch_size is not None else patch_size
        if dim_index is None:
            dim_index = list(range(len(size) - len(patch_size), len(size)))
        for i, dim in enumerate(dim_index):
            factor = size[dim] // patch_size[i]
            if factor != 1:
                return factor
        return 1
