from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List
from .utils import *
from .patch import *
import concurrent.futures
import threading
import torch
from torch import Tensor, dtype


@dataclass
class State:
    """Holds the current state of training process"""
    shape: List[int]
    tensor: Tensor
    data_dtype: dtype
    orig_size: List[int]
    tuple_indices: List[int]
    code_list: List[int]
    code_indices: List[int]
    joined_list: List[int]


class BaseTokenizer:
    """Base class for Tokenizer"""

    def __init__(self):
        self.vocab = dict()
        self.inverse_vocab = dict()

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def train(self, data, max_shape, dim_index, min_freq, root_min_freq):
        # Tokenizer can train a vocabulary of size vocab_size from given data
        raise NotImplementedError

    def encode(self, data, max_shape):
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

    def train(self, data_loader, max_shape, dim_index, min_freq=2, root_min_freq=2):
        """
        Train a vocabulary of using the provided data.

        Args:
            data (array-like): The input data that is shuffled into a list of tuples.
            max_shape (tuple): The maximum shape of the tuples.
            min_freq (int): The minimum frequency of a pair to be considered.
            root_min_freq (int): The minimum frequency of a pair to be considered for the root vocabulary.

        Returns:
            None
        """
        # if len(data.size()) != 4:
        #     raise ValueError(f"Expected 4D input tensor, got shape {data.size()}")

        shapes = find_tuple_shapes(max_shape)

        # Initialize a dictionary to store states for all images
        states = {}

        # create initial states for all images
        for batch_idx, (data, _) in enumerate(data_loader): # TODO: need to check data loader shuffle
            data = data.unsqueeze(
                1)  # create sequence dimension # TODO: add to transform, assuming sequence length here
            for i in range(len(data)):
                data_i = data[[i]]
                state_key = f"batch_{batch_idx}_{i}"  ## TODO: this needs discussion, should be set in thread
                states[state_key] = State(
                    shape=[],
                    tensor=data_i,
                    data_dtype=data_i.dtype,
                    orig_size=list(data_i.size()),
                    tuple_indices=[],
                    code_list=[],
                    code_indices=[],
                    joined_list=[]
                )

        # TODO: should each batch set a thread
        # Process each shape for all images
        for shape in shapes:
            if self.max_workers is not None and self.max_workers > 0:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._train_single,
                            state,
                            shape,
                            dim_index,
                            min_freq,
                            root_min_freq
                        ) for state in states.values()
                    ]

                    # Update states with results
                    for future, key in zip(concurrent.futures.as_completed(futures), states.keys()):
                        updated_state = future.result()
                        if updated_state is not None:
                            states[key] = updated_state
            else:
                for key, state in states.items():
                    updated_state = self._train_single(
                        state,
                        shape,
                        dim_index,
                        min_freq,
                        root_min_freq
                    )
                    if updated_state is not None:
                        states[key] = updated_state

        return

    def _train_single(self, state, shape, dim_index, min_freq, root_min_freq):
        """Process a single batch of data"""
        state.shape = shape
        patchify = Patchify(shape, dim_index)

        # Split data if there is joined data
        scale_factor = patchify.get_scale_factor(state.orig_size)
        if len(state.joined_list) > 0:
            state = self._split_data(state, scale_factor)

        if state is not None:
            # Process root vocabulary
            state = self._process_root_vocabulary(state, patchify, root_min_freq)
            tuple_list = tensor_to_tuple(state.tensor, state.shape)[0]
            # Join back for merging
            joined_list = join(tuple_list, state.tuple_indices, state.code_list, state.code_indices)
            state.joined_list = self._merge_pairs(joined_list, min_freq)
        return state

    def _process_root_vocabulary(self, state, patchify, root_min_freq):
        """Process root vocabulary for a single shape configuration"""
        unshuffled_tensor = patchify(state.tensor)
        state.orig_size = list(unshuffled_tensor.size())  # Update state for next iteration

        # Determine minimum frequency for root vocabulary
        # TODO: if state.shape == [1, 1, 1] is not generic enough, should check for more than 3 dimensions
        current_root_min_freq = 1 if state.shape == [1, 1, 1] else root_min_freq

        tensor = unshuffled_tensor[0]
        output, inverse_indices, counts = torch.unique(tensor, return_inverse=True, return_counts=True, dim=0)

        # Find frequent patterns and assign codes
        freq_mask = counts >= current_root_min_freq  # TODO: this needs to be tracked online
        root_tuples = tensor_to_tuple(output[freq_mask].unsqueeze(0), state.shape)[0]

        with self._lock:
            unique_codes = self._assign_codes(root_tuples)

        # Create mapping for all patterns
        full_mapping = torch.full((counts.size(0),), -1, dtype=torch.long)
        full_mapping[freq_mask] = torch.tensor(unique_codes, dtype=torch.long)
        mapped_values = full_mapping[inverse_indices]

        # Split indices
        non_root_indices = torch.nonzero(mapped_values == -1).squeeze()  # TODO: maybe using torch.where instead
        code_indices = torch.nonzero(mapped_values != -1).squeeze()
        codes = mapped_values[code_indices]

        # Convert to lists for further processing
        non_root_indices = np.atleast_1d(non_root_indices).tolist() # TODO: do not use numpy for pytorch tensor
        codes = codes.tolist()
        code_indices = code_indices.tolist()

        # Update tensor with non-root indices
        new_tensor = tensor[non_root_indices].unsqueeze(0)

        # Update tuple indices if they exist
        # TODO: what is this for?
        if len(state.tuple_indices) > 0:
            code_indices = torch.tensor(state.tuple_indices)[code_indices].tolist()
            non_root_indices = torch.tensor(state.tuple_indices)[non_root_indices].tolist()

        # TODO: is it possible to just update the instance instead of creating a new one everytime you return?
        state = State(
            shape=state.shape,
            tensor=new_tensor,
            data_dtype=state.data_dtype,
            orig_size=state.orig_size,
            tuple_indices=non_root_indices,
            code_list=state.code_list + list(map(str, codes)),
            code_indices=state.code_indices + code_indices,
            joined_list=state.joined_list
        )
        return state

    def _split_data(self, state, scale_factor):
        """Split data into tuple and code lists"""
        tuple_list, tuple_indices, code_list, code_indices = split(state.joined_list, scale_factor)

        if len(tuple_list) > 0:
            orig_size = state.orig_size.copy()
            orig_size[1] = len(tuple_list)  # Update the length dimension
            tensor = tuple_to_tensor(tuple_list, state.shape, orig_size, state.data_dtype)
            state = State(
                shape=state.shape,
                tensor=tensor,
                data_dtype=state.data_dtype,
                orig_size=orig_size,
                tuple_indices=tuple_indices,
                code_list=code_list,
                code_indices=code_indices,
                joined_list=state.joined_list
            )
        else:
            state = None
        return state

    def _assign_codes(self, tuples):
        """Assign codes to the root tuples and update the vocabulary"""
        # Assign existing codes
        unique_codes = np.array([int(self.inverse_vocab.get(tup, -1)) for tup in tuples])

        # Identify new codes
        new_codes_mask = unique_codes == -1
        new_tuples = np.fromiter(tuples, dtype=object)[new_codes_mask]
        new_indices = np.arange(len(self.vocab), len(self.vocab) + len(new_tuples))

        # Assign new codes
        unique_codes[new_codes_mask] = new_indices
        unique_codes_str = list(map(str, new_indices))
        update_vocab(self.vocab, self.inverse_vocab, new_tuples, unique_codes_str)
        return unique_codes

    def _merge_pairs(self, data, min_freq):
        """Merge pairs in the data"""
        while True:
            stats = get_freq_pairs(data)
            pair, freq = get_max_pair(stats)

            if freq < min_freq:  # this min_freq needs to be tracked online
                break

            with self._lock:
                # Look up the pair in the inverse_vocab
                code = self.inverse_vocab.get(pair, None)
                if code is None:
                    idx = str(len(self.vocab))
                    update_vocab(self.vocab, self.inverse_vocab, pair, idx)
                else:
                    idx = code

            data = merge(data, pair, idx)
        return data

    def encode(self, data, max_shape):
        """
        Encode using the trained vocabulary.

        Args:
            data (array-like): The input data that is shuffled into a list of tuples.
            max_shape (tuple): The initial dimension for the tuples.

        Returns:
            tuple_list (list): The encoded list of tuples.
        """

        if len(self.vocab) == 0:
            raise ValueError('Vocabulary not trained yet.')

        tuple_list = data
        shapes = find_tuple_shapes(max_shape)

        for i in range(len(shapes)):
            # the input is already reshaped so we skip reshaping in the first iteration
            if i > 0:
                tuple_list = tuple_reshape(tuple_list, shapes[i - 1], shapes[i])

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
            encoded (list): encoded data containing string codes.

        Returns:
            plain_list (list): The decoded list of tuples.
        """

        decoded = []
        for i in encoded:
            pair = self.vocab[i]
            decoded.append(dfs(pair, self.vocab))

        return np.concatenate(decoded).tolist()
