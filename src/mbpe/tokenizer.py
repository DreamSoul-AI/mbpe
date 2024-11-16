from dataclasses import dataclass
from typing import List
from collections import defaultdict
from .utils import *
import torch
from torch import Tensor, dtype


@dataclass
class State:
    """Holds the current state of training process"""
    shape: List[int]
    data: Tensor
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
    def __init__(self):
        super().__init__()

    def train(self, data, max_shape, min_freq=2, root_min_freq=2):
        """
        Train a vocabulary of using the provided data.

        Args:
        - data (array-like): The input data that is shuffled into a list of tuples.
        - dim (tuple): The initial dimension for the tuples.
        - min_freq (int): The minimum frequency of a pair to be considered.
        - root_min_freq (int): The minimum frequency of a pair to be considered for the root vocabulary.

        Returns:
        - None
        """

        state = State(
            shape=[],
            data=data,
            data_dtype=data.dtype,
            orig_size=list(data.size()),
            tuple_indices=[],
            code_list=[],
            code_indices=[]
        )

        shapes = find_tuple_shapes(max_shape)

        for shape in shapes:
            state.shape = shape
            state = self._process_shape(state, min_freq, root_min_freq)

        return

    def encode(self, data, dim):
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

        tuple_list = data
        shapes = find_tuple_shapes(dim)

        for i in range(len(shapes)):
            # the input is already reshaped so we skip reshaping in the first iteration
            if i > 0:
                tuple_list = tuple_reshape(tuple_list, shapes[i-1], shapes[i])

            # update with the root vocabulary
            for i, t in enumerate(tuple_list):
                if t in self.inverse_vocab.keys():
                    tuple_list[i] = self.inverse_vocab[t]

            # merge pairs
            while True:
                stats = get_freq_pairs(tuple_list)
                pair, _ = get_max_pair(stats)

                if pair not in self.inverse_vocab.keys():
                    break

                # look up the pair in the inverse_vocab
                idx = self.inverse_vocab[pair]

                tuple_list = merge(tuple_list, pair, idx)

        return tuple_list

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

    def _process_shape(self, state, min_freq, root_min_freq) -> State:
        """Process data for a single shape configuration"""
        n = len(state.orig_size)
        dim_index = list(range(n - 3, n))  # Last 3 dimensions

        # Calculate scale factors
        scale_factor = [
            state.orig_size[dim_index[i]] // state.shape[i]
            for i in range(len(dim_index))
        ]

        # Split data if there are existing codes
        if len(state.code_list) > 0:
            state = self._split_data(state, scale_factor)

        unshuffled_tensor = tensor_unshuffle(state.data, scale_factor, dim_index)
        state.orig_size = list(unshuffled_tensor.size())    # Update state for next iteration

        # Handle root vocabulary
        current_root_min_freq = 1 if state.shape == [1, 1, 1] else root_min_freq
        state = self._update_root_vocabulary(state, unshuffled_tensor, current_root_min_freq)

        # Join the data back for merging
        state.data = join(state.data, state.code_list, state.code_indices)
        # Merge vocabulary pairs
        state = self._merge_pairs(state, min_freq)

        return state

    def _split_data(self, state, scale_factor) -> State:
        """Handle processing of existing codes"""
        data, tuple_indices, code_list, code_indices = split(state.data, list(set(scale_factor))[-1])
        orig_size = state.orig_size.copy()
        orig_size[1] = len(data)  # Update the length dimension

        return State(
            shape=state.shape,
            data=tuple_to_tensor(data, state.shape, orig_size, state.data_dtype),
            data_dtype=state.data_dtype,
            orig_size=orig_size,
            tuple_indices=tuple_indices,
            code_list=code_list,
            code_indices=code_indices
        )

    def _update_root_vocabulary(self, state, tensor, root_min_freq) -> State:
        """Update vocabulary with new patterns"""
        for i in range(len(tensor)):
            tensor_i = tensor[i]
            root, root_indices, unique_codes, codes, indices = find_root(tensor_i, root_min_freq, self.vocab)

            # Update with root vocabulary
            root_tuples = tensor_to_tuple(root.unsqueeze(0), state.shape)[0]
            update_vocab(self.vocab, self.inverse_vocab, root_tuples, list(map(str, unique_codes)))

            # Process tuple indices
            tuple_list = tensor_to_tuple(tensor_i[root_indices].unsqueeze(0), state.shape)[0]

            # Update current code indices
            if len(state.tuple_indices) > 0:
                indices = torch.tensor(state.tuple_indices)[indices].tolist()
            new_code_list = state.code_list + list(map(str, codes))
            new_code_indices = state.code_indices + indices

        return State(
            shape=state.shape,
            data=tuple_list,
            data_dtype=state.data_dtype,
            orig_size=state.orig_size,
            tuple_indices=state.tuple_indices,
            code_list=new_code_list,
            code_indices=new_code_indices
        )

    def _merge_pairs(self, state, min_freq) -> State:
        print(state.data)
        # TODO: need to consider batch size
        """Merge pairs in the data"""
        while True:
            stats = get_freq_pairs(state.data)
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

            state.data = merge(state.data, pair, idx)

        return state
