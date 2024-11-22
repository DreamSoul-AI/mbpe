from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List
from collections import defaultdict
from .utils import *
import concurrent.futures
import threading
import os
import torch
from torch import Tensor, dtype


@dataclass
class State:
    """Holds the current state of training process"""
    shape: List[int]
    tuples: Tensor
    data_dtype: dtype
    orig_size: List[int]
    tuple_indices: List[int]
    code_list: List[int]
    code_indices: List[int]


class BaseTokenizer:
    """Base class for Tokenizer"""

    def __init__(self):
        self.vocab = defaultdict(tuple)
        self.inverse_vocab = defaultdict(str)

    def get_vocab(self):
        return self.vocab

    def get_vocab_len(self):
        return len(self.vocab)

    def train(self, data, dim, min_freq):
        # Tokenizer can train a vocabulary of size vocab_size from given data
        raise NotImplementedError

    def encode(self, data, dim):
        # Tokenizer can encode a list of tuples based on the trained vocabulary
        raise NotImplementedError

    def decode(self, encoded):
        # Tokenizer can decode a list of encoded tuples into the original data
        raise NotImplementedError


class Tokenizer(BaseTokenizer):
    def __init__(self, max_workers=None):
        super().__init__()
        self.max_workers = max_workers
        self._lock = threading.Lock()

    def _train_single_batch(self, batch_data, max_shape, min_freq, root_min_freq):
        """Process a single batch of data"""
        state = State(
            shape=[],
            tuples=batch_data,
            data_dtype=batch_data.dtype,
            orig_size=list(batch_data.size()),
            tuple_indices=[],
            code_list=[],
            code_indices=[]
        )

        shapes = find_tuple_shapes(max_shape)
        joined_list = []
        for shape in shapes:
            state.shape = shape
            state = self._process_root(state, joined_list, root_min_freq)
            # Join back for merging
            joined_list = join(state.tuples, state.code_list, state.code_indices)

            # Thread-safe vocabulary updates
            with self._lock:
                joined_list = self._merge_pairs(joined_list, min_freq)

        return joined_list

    def train(self, data, max_shape, min_freq=2, root_min_freq=2):
        """
        Train a vocabulary of using the provided data.

        Args:
        - data (array-like): The input data that is shuffled into a list of tuples.
        - max_shape (tuple): The maximum shape of the tuples.
        - min_freq (int): The minimum frequency of a pair to be considered.
        - root_min_freq (int): The minimum frequency of a pair to be considered for the root vocabulary.

        Returns:
        - None
        """
        if len(data.size()) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {data.size()}")

        # Create a list of individual batch tensors
        batch_tensors = [data[i].unsqueeze(0) for i in range(data.size(0))]

        # Process batches concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(
                    self._train_single_batch,
                    batch,
                    max_shape,
                    min_freq,
                    root_min_freq
                ): i for i, batch in enumerate(batch_tensors)
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    joined_list = future.result()
                    print(f'Batch {batch_idx} processed successfully.')
                    print(joined_list)
                except Exception as e:
                    print(f'Batch {batch_idx} generated an exception: {e}')
        return

    def encode(self, data, max_shape):
        """
        Encode using the trained vocabulary.

        Args:
        - data (array-like): The input data that is shuffled into a list of tuples.
        - dim (tuple): The initial dimension for the tuples.

        Returns:
        - tuple_list (list): The encoded list of tuples.
        """

        if len(self.vocab) == 0:
            raise ValueError('Vocabulary not trained yet.')

        shapes = find_tuple_shapes(max_shape)

        joined_list = []
        for shape in shapes:
            n = len(state.orig_size)
            dim_index = list(range(n - 3, n))  # Last 3 dimensions

            # Calculate scale factors
            scale_factor = [
                state.orig_size[dim_index[i]] // state.shape[i]
                for i in range(len(dim_index))
            ]

            # Split data if there is joined data
            if len(joined_list) > 0:
                state = self._split_data(state, joined_list, scale_factor)

            unshuffled_tensor = tensor_unshuffle(state.tuples, scale_factor, dim_index)
            # Join back for merging
            joined_list = join(state.tuples, state.code_list, state.code_indices)
            # Merge vocabulary pairs
            joined_list = self._merge_pairs(joined_list)

        return joined_list

        # tuple_list = data
        # shapes = find_tuple_shapes(dim)

        # for i in range(len(shapes)):
        #     # the input is already reshaped so we skip reshaping in the first iteration
        #     if i > 0:
        #         tuple_list = tuple_reshape(tuple_list, shapes[i-1], shapes[i])

        #     # update with the root vocabulary
        #     for i, t in enumerate(tuple_list):
        #         if t in self.inverse_vocab.keys():
        #             tuple_list[i] = self.inverse_vocab[t]

        #     # merge pairs
        #     while True:
        #         stats = get_freq_pairs(tuple_list)
        #         pair, _ = get_max_pair(stats)

        #         if pair not in self.inverse_vocab.keys():
        #             break

        #         # look up the pair in the inverse_vocab
        #         idx = self.inverse_vocab[pair]

        #         tuple_list = merge(tuple_list, pair, idx)

        # return tuple_list

    def decode(self, encoded):
        """
        Decode the encoded data back into its original list of tuples.

        Args:
        - encoded (list): encoded data containing tuples and string codes.

        Returns:
        - plain_list (list): The decoded list of tuples.
        """

        decoded = []
        for i in encoded:
            pair = self.vocab[i]
            decoded.append(dfs(pair, self.vocab))
        plain_list = [item for tup in decoded for item in tup]

        return plain_list

    def _process_root(self, state, joined_list, root_min_freq) -> State:
        """Process root vocabulary for a single shape configuration"""
        n = len(state.orig_size)
        dim_index = list(range(n - 3, n))  # Last 3 dimensions

        # Calculate scale factors
        scale_factor = [
            state.orig_size[dim_index[i]] // state.shape[i]
            for i in range(len(dim_index))
        ]

        # Split data if there is joined data
        if len(joined_list) > 0:
            state = self._split_data(state, joined_list, scale_factor)

        unshuffled_tensor = tensor_unshuffle(state.tuples, scale_factor, dim_index)
        state.orig_size = list(unshuffled_tensor.size())    # Update state for next iteration

        # Handle root vocabulary
        current_root_min_freq = 1 if state.shape == [1, 1, 1] else root_min_freq
        with self._lock:
            state = self._update_root_vocabulary(state, unshuffled_tensor[0], current_root_min_freq)

        return state

    def _split_data(self, state, joined_list, scale_factor) -> State:
        """Split data into tuple and code lists"""
        tuple_list, tuple_indices, code_list, code_indices = split(joined_list, list(set(scale_factor))[-1])
        orig_size = state.orig_size.copy()
        orig_size[1] = len(tuple_list)  # Update the length dimension

        return State(
            shape=state.shape,
            tuples=tuple_to_tensor(tuple_list, state.shape, orig_size, state.data_dtype),
            data_dtype=state.data_dtype,
            orig_size=orig_size,
            tuple_indices=tuple_indices,
            code_list=code_list,
            code_indices=code_indices
        )

    def _update_root_vocabulary(self, state, tensor, root_min_freq) -> State:
        """Update vocabulary with new patterns"""
        root, root_indices, unique_codes, codes, indices = find_root(tensor, root_min_freq, self.vocab)

        # Update with root vocabulary
        root_tuples = tensor_to_tuple(root.unsqueeze(0), state.shape)[0]
        update_vocab(self.vocab, self.inverse_vocab, root_tuples, list(map(str, unique_codes)))

        # Process tuple indices
        tuple_list = tensor_to_tuple(tensor[root_indices].unsqueeze(0), state.shape)[0]

        # Update current code indices
        if len(state.tuple_indices) > 0:
            indices = torch.tensor(state.tuple_indices)[indices].tolist()
        new_code_list = state.code_list + list(map(str, codes))
        new_code_indices = state.code_indices + indices

        return State(
            shape=state.shape,
            tuples=tuple_list,
            data_dtype=state.data_dtype,
            orig_size=state.orig_size,
            tuple_indices=state.tuple_indices,
            code_list=new_code_list,
            code_indices=new_code_indices
        )

    def _merge_pairs(self, data, min_freq):
        """Merge pairs in the data"""
        while True:
            stats = get_freq_pairs(data)
            pair, freq = get_max_pair(stats)

            if freq < min_freq:
                break

            # Look up the pair in the inverse_vocab
            code = self.inverse_vocab[pair]
            if code != '':
                idx = code
            else:
                idx = str(len(self.vocab))
                update_vocab(self.vocab, self.inverse_vocab, pair, idx)

            data = merge(data, pair, idx)

        return data
