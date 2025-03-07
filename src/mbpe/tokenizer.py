from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List
from .utils import *
from .patch import *
from .frequency_counter import FrequencyCounter
import concurrent.futures
import threading
import torch
from torch import Tensor, dtype
from torch.utils.data import DataLoader


@dataclass
class State: # TODO: make a standard class
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

    def train(self, data, data_name, max_shape, dim_index, min_freq, root_min_freq, min_entrance_freq):
        # Tokenizer can train a vocabulary of size vocab_size from given data
        raise NotImplementedError

    def encode(self, data, max_shape):
        # Tokenizer can encode a list of tuples based on the trained vocabulary
        raise NotImplementedError

    def decode(self, encoded):
        # Tokenizer can decode a list of encoded tuples into the original data
        raise NotImplementedError


class Tokenizer(BaseTokenizer):
    def __init__(self, batch_size=1, max_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._lock = threading.Lock()

    def train(self, dataset, data_name, max_shape, dim_index, min_freq, root_min_freq, min_entrance_freq):
        # TODO: make the data loader inside to make sure shuffle = False
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        shapes = find_tuple_shapes(max_shape)
        print('Found Shapes:{}'.format(shapes))
        # Initialize a dictionary to store states for all images
        states = {}

        # TODO: rewrite
        for shape in shapes:
            patchify = Patchify(shape, dim_index)
            freq_counter = FrequencyCounter(min_entrance_freq, root_min_freq)
            index_tracker = 0

            for batch_idx, data in enumerate(data_loader):
                data = data[data_name]
                for i in range(len(data)):
                    data_i = data[[i]]
                    if index_tracker not in states:
                        state_i = State(
                            shape=shape,
                            tensor=data_i,
                            data_dtype=data_i.dtype,
                            orig_size=list(data_i.size()),
                            tuple_indices=[],
                            code_list=[],
                            code_indices=[],
                            joined_list=[]
                        )
                        states[index_tracker] = state_i
                        pachified_data = patchify(data_i)[0]
                        freq_counter.update_freq_tables(pachified_data, self.vocab, self.inverse_vocab)
                    else:
                        state_i = states[index_tracker]
                        state_i.shape = shape
                        pachified_data = patchify(state_i.tensor)[0]
                        freq_counter.update_freq_tables(pachified_data, self.vocab, self.inverse_vocab)
                    updated_state = self._process_batch_root(state_i, patchify)
                    if updated_state is not None: # TODO: do we need to check
                        states[index_tracker] = updated_state


        # # Process each shape for all images
        # for shape in shapes:
        #     patchify = Patchify(shape, dim_index)
        #     counter = FrequencyCounter(min_entrance_freq, root_min_freq)
        #     self._setup(data_loader, shape, patchify, counter, states) # What is this for?
        #
        #     print(self.vocab)
        #     print(counter)
        #
        #     # Process the batch using threading
        #     for _, batch_states in states.items():
        #         # if self.max_workers is not None and self.max_workers > 0:
        #         #     with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        #         #         futures = [executor.submit(self._process_batch_root, state, patchify) for state in batch_states]
        #         #
        #         #         # Update batch states with results
        #         #         for i, future in enumerate(concurrent.futures.as_completed(futures)):
        #         #             updated_state = future.result()
        #         #             if updated_state is not None:
        #         #                 batch_states[i] = updated_state
        #         # else:
        #         for i in range(len(batch_states)): # TODO: why for loop batch again?
        #             updated_state = self._process_batch_root(batch_states[i], patchify)
        #             if updated_state is not None:
        #                 batch_states[i] = updated_state

            # Merge pairs
            self._merge_pairs(states, min_freq, min_entrance_freq) # merge with loop before # TODO: create freq outside
        return

    # def _setup(self, data_loader, shape, patchify, counter, states): # TODO: name should change, should merge with next loop
    #     for batch_idx, (image, _) in enumerate(data_loader):
    #         batch_state_key = f"batch_{batch_idx}" # TODO: this is wrong, should index with batch
    #         if batch_state_key not in states.keys():
    #             batch_states = []
    #             for i in range(len(image)):
    #                 image_i = image[[i]]
    #                 state_i = State(
    #                     shape=shape,
    #                     tensor=image_i,
    #                     data_dtype=image_i.dtype,
    #                     orig_size=list(image_i.size()),
    #                     tuple_indices=[],
    #                     code_list=[],
    #                     code_indices=[],
    #                     joined_list=[]
    #                 )
    #                 batch_states.append(state_i)
    #                 pachified_image = patchify(image_i)[0]
    #                 counter.update_freq_tables(pachified_image, self.vocab, self.inverse_vocab)
    #         else:
    #             batch_states = states[batch_state_key]
    #             for state in batch_states:
    #                 state.shape = shape
    #                 pachified_image = patchify(state.tensor)[0]
    #                 counter.update_freq_tables(pachified_image, self.vocab, self.inverse_vocab)
    #
    #         states[batch_state_key] = batch_states
    #     return

    def _process_batch_root(self, state, patchify): # TODO: need to rename
        """Process a single batch of data"""
        # Split data if there is joined data
        scale_factor = patchify.get_scale_factor(state.orig_size)
        if len(state.joined_list) > 0:
            state = self._split_data(state, scale_factor)

        if state is not None:
            # Process root vocabulary
            state = self._process_root_state(state, patchify)
            tuple_list = tensor_to_tuple(state.tensor, state.shape)[0]
            # Join back for merging
            state.joined_list = join(tuple_list, state.tuple_indices, state.code_list, state.code_indices)
        return state

    def _process_root_state(self, state, patchify):
        """Process root state for a single shape configuration"""
        unshuffled_tensor = patchify(state.tensor)
        state.orig_size = list(unshuffled_tensor.size())  # Update state for next iteration

        # Determine if this is the last iteration
        last_shape = True if all(dim == 1 for dim in state.shape) else False

        tensor = unshuffled_tensor[0]
        tuple_list = tensor_to_tuple(unshuffled_tensor, state.shape)[0]
        code_mapping = torch.full((tensor.size(0),), -1, dtype=torch.long)
        # Update code_mapping using the root vocabulary
        with self._lock:
            for i, tup in enumerate(tuple_list):
                code = self.inverse_vocab.get(tup, None)
                if code is not None:
                    code_mapping[i] = int(code)
                else:
                    if last_shape:
                        code_mapping[i] = len(self.vocab)
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

    def _merge_pairs(self, states, min_freq, min_entrance_freq): # TODO: this should take in state not states
        while True:
            counter = FrequencyCounter(min_entrance_freq)  # TODO: why another counter here?
            # for batch_states in states.values(): # TODO: Another for loop here
            #     for state in batch_states:
            #         counter.update_merge_freq_tables(state.joined_list)
            for state_i in states.values():
                counter.update_merge_freq_tables(state_i.joined_list)

            freq_table = counter.get_global_freq_table()
            # print(freq_table)
            if len(freq_table) == 0:
                break

            pair, freq = counter.get_max_merge_pair().values()
            print(pair, freq)
            if freq < min_freq:
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
                    # print(state.joined_list)

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
