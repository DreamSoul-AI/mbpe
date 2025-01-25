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

    def train(self, data_loader, max_shape, dim_index, min_freq, root_min_freq, min_entrance_freq=2):
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

        # Process each shape for all images
        for shape in shapes:
            batch_state_key = ""
            patchify = Patchify(shape, dim_index)
            total_tuple_count = 0
            freq_table = defaultdict(int)

            # Compute frequencies of roots for all data
            for batch_idx, (image, _) in enumerate(data_loader):
                batch_state_key = f"batch_{batch_idx}"
                if batch_state_key not in states.keys():
                    batch_states = []
                    for i in range(len(image)):
                        image_i = image[[i]]
                        state_i = State(
                            shape=shape,
                            tensor=image_i,
                            data_dtype=image_i.dtype,
                            orig_size=list(image_i.size()),
                            tuple_indices=[],
                            code_list=[],
                            code_indices=[],
                            joined_list=[]
                        )
                        batch_states.append(state_i)
                        pachified_image = patchify(image_i)[0]
                        freq_table = compute_freq(pachified_image, freq_table, min_entrance_freq)
                        total_tuple_count += len(pachified_image)
                else:
                    batch_states = states[batch_state_key]
                    for state in batch_states:
                        state.shape = shape
                        pachified_tensor = patchify(state.tensor)[0]
                        freq_table = compute_freq(pachified_image, freq_table, min_entrance_freq)
                        total_tuple_count += len(pachified_tensor)

                states[batch_state_key] = batch_states

            # Filter the frequency table and update root vocabulary
            for tuple_key in freq_table:
                freq = freq_table[tuple_key] / total_tuple_count
                if freq >= root_min_freq:
                    update_vocab(self.vocab, self.inverse_vocab, tuple_key, str(len(self.vocab)))

            print(self.vocab)

            # Process the batch using threading
            for _, batch_states in states.items():
                if self.max_workers is not None and self.max_workers > 0:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = [
                            executor.submit(
                                self._train_single,
                                state,
                                patchify,
                                min_freq
                            ) for state in batch_states
                        ]

                        # Update batch states with results
                        for i, future in enumerate(concurrent.futures.as_completed(futures)):
                            updated_state = future.result()
                            if updated_state is not None:
                                batch_states[i] = updated_state
                else:
                    for i in range(len(batch_states)):
                        updated_state = self._train_single(batch_states[i], patchify, min_freq)
                        if updated_state is not None:
                            batch_states[i] = updated_state
                break

            while True:
                freq_table = defaultdict(int)
                total_pair_count = 0
                for batch_states in states.values():
                    for state in batch_states:
                        freq_table = get_freq_pairs(state.joined_list, freq_table, min_entrance_freq)
                        total_pair_count += len(state.joined_list)
                if len(freq_table) == 0:
                    break
                pair, freq = get_max_pair(freq_table)
                print(freq_table)
                print(freq / total_pair_count)
                if freq / total_pair_count < min_freq:
                    break
                code = self.inverse_vocab.get(pair, None)
                if code is None:
                    idx = str(len(self.vocab))
                    update_vocab(self.vocab, self.inverse_vocab, pair, idx)
                else:
                    idx = code
                for batch_states in states.values():
                    for state in batch_states:
                        state.joined_list = merge(state.joined_list, pair, idx)
                        print(state.joined_list)
                        break
            break

        return

    def _train_single(self, state, patchify, min_freq):
        """Process a single batch of data"""
        # Split data if there is joined data
        scale_factor = patchify.get_scale_factor(state.orig_size)
        if len(state.joined_list) > 0:
            state = self._split_data(state, scale_factor)

        if state is not None:
            # Process root vocabulary
            state = self._process_root_vocabulary(state, patchify)
            tuple_list = tensor_to_tuple(state.tensor, state.shape)[0]
            # Join back for merging
            state.joined_list = join(tuple_list, state.tuple_indices, state.code_list, state.code_indices)
            # state.joined_list = self._merge_pairs(joined_list, min_freq)
        return state

    def _process_root_vocabulary(self, state, patchify):
        """Process root vocabulary for a single shape configuration"""
        unshuffled_tensor = patchify(state.tensor)
        state.orig_size = list(unshuffled_tensor.size())  # Update state for next iteration

        # Determine if this is the last iteration
        last_shape = True if all(dim == 1 for dim in state.shape) else False

        tensor = unshuffled_tensor[0]
        code_mapping = torch.full((tensor.size(0),), -1, dtype=torch.long)
        with self._lock:
            for i, item in enumerate(tensor):
                tup = tuple(item.flatten().tolist())
                code = self.inverse_vocab.get(tup, None)
                if code is not None:
                    code_mapping[i] = int(code)
                else:
                    if last_shape:
                        update_vocab(self.vocab, self.inverse_vocab, tup, len(self.vocab))

        non_root_indices = torch.where(code_mapping == -1)[0]
        root_indices = torch.where(code_mapping != -1)[0]
        root_codes = code_mapping[root_indices].tolist()
        state.tensor = tensor[non_root_indices].unsqueeze(0)
        state.code_list.extend(list(map(str, root_codes)))

        if len(state.tuple_indices) > 0:
            root_indices = torch.tensor(state.tuple_indices)[root_indices].tolist()
            non_root_indices = torch.tensor(state.tuple_indices)[non_root_indices].tolist()
        else:
            root_indices = root_indices.tolist()
            non_root_indices = non_root_indices.tolist()

        # Update state for current shape
        state.tuple_indices = non_root_indices
        state.code_indices.extend(root_indices)

        return state

    def _split_data(self, state, scale_factor):
        """Split data into tuple and code lists"""
        tuple_list, tuple_indices, code_list, code_indices = split(state.joined_list, scale_factor)

        if len(tuple_list) > 0:
            state.orig_size[1] = len(tuple_list)  # Update the length dimension
            state.tensor = tuple_to_tensor(tuple_list, state.shape, state.orig_size, state.data_dtype)
            state.tuple_indices = tuple_indices
            state.code_list = code_list
            state.code_indices = code_indices
            return state

        return None

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
