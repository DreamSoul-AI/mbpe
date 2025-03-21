import concurrent.futures
import threading
import torch
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from .frequency_counter import FrequencyCounter
from .utils import *
from .patch import *
from .state import State


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

    @torch.no_grad()
    def train(self, dataset, data_name, max_code_size, dim_index, min_freq, root_min_freq, min_entrance_freq):
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        code_sizes = find_tuple_shapes(max_code_size)
        print('Found Code Sizes:{}'.format(code_sizes))
        states = {}
        # TODO: rewrite (working)
        for m, code_size in enumerate(code_sizes):
            patchify = Patchify(code_size, dim_index)
            # freq_counter = FrequencyCounter(min_entrance_freq, root_min_freq)
            index_tracker = 0
            for batch_idx, data in enumerate(data_loader):
                data = data[data_name]
                for i in range(len(data)):  # TODO: add multithread/multiprocess here (later)
                    print(i)
                    data_i = data[i]
                    if index_tracker not in states:
                        states[index_tracker] = State(data_i, code_size)
                    # state_i = states[index_tracker]
                    # state_i.code_size = code_size
                    # patch_data = patchify(state_i.data, is_batch=False)
                    # freq_counter.update_freq_tables(patch_data, self.vocab, self.inverse_vocab)
                    is_last = m == len(code_sizes) - 1
                    states[index_tracker] = self.process_root(states[index_tracker], patchify, is_last)
                    # updated_state = self.process_root(state_i, patchify, is_last)
                    # if updated_state is not None:  # TODO: do we need?
                    #     states[index_tracker] = updated_state
                    self._merge_pairs(states[index_tracker], min_freq, min_entrance_freq)
                    print(states[index_tracker])
            # Merge pairs
            # self._merge_pairs(states, min_freq, min_entrance_freq)  # merge with loop before # TODO: create freq outside
        return

    def process_root(self, state, patchify, is_last):
        if len(state.joined_list) > 0: # TODO: this should be moved to merge
            scale_factor = patchify.get_scale_factor(state.size)
            tuple_list, tuple_indices, code_list, code_indices = split(state.joined_list, scale_factor)
            if len(tuple_list) > 0:
                state.size[0] = len(tuple_list)  # Update the length dimension
                state.data = tuple_to_tensor(tuple_list, state.code_size, state.size, state.dtype, is_batch=False)
                state.tuple_list = tuple_list
                state.tuple_indices = tuple_indices
                state.code_list = code_list
                state.code_indices = code_indices
        else:  # TODO: this only runs for the first shape, when the joined list not exist?
            unshuffled_data = patchify(state.data, is_batch=False)
            state.size = list(unshuffled_data.size()) # TODO: duplicate
            # TODO: this could be run initialily
            tuple_list = tensor_to_tuple(unshuffled_data, state.code_size, is_batch=False)
            code_mapping = torch.full((unshuffled_data.size(0),), -1, dtype=torch.long)
            for i, tup in enumerate(tuple_list):
                code = self.inverse_vocab.get(tup, None)
                if code is not None:
                    code_mapping[i] = int(code)  # TODO: need min freq here
                else:
                    if is_last:
                        code_mapping[i] = len(self.vocab)
                        update_vocab(self.vocab, self.inverse_vocab, tup, len(self.vocab))
            non_root_indices = torch.where(code_mapping == -1)[0]
            root_indices = torch.where(code_mapping != -1)[0]
            root_codes = code_mapping[root_indices].tolist()
            state.data = unshuffled_data[non_root_indices]
            state.code_list.extend(list(map(str, root_codes)))
            if len(state.tuple_indices) > 0:
                root_indices = torch.tensor(state.tuple_indices)[root_indices].tolist()
                non_root_indices = torch.tensor(state.tuple_indices)[non_root_indices].tolist()
            else:
                root_indices = root_indices.tolist()
                non_root_indices = non_root_indices.tolist()
            state.tuple_indices = non_root_indices
            state.code_indices.extend(root_indices)
            tuple_list = tensor_to_tuple(state.data, state.code_size, is_batch=False)
            # TODO: has bug
            state.joined_list = join(tuple_list, state.tuple_indices, state.code_list, state.code_indices)
        return state

    # def _process_batch_root(self, state, patchify):
    #     """Process a single batch of data"""
    #     # Split data if there is joined data
    #     scale_factor = patchify.get_scale_factor(state.orig_size)
    #     if len(state.joined_list) > 0:
    #         state = self._split_data(state, scale_factor)
    #
    #     if state is not None:
    #         # Process root vocabulary
    #         state = self._process_root_state(state, patchify)
    #         tuple_list = tensor_to_tuple(state.tensor, state.shape)[0]
    #         # Join back for merging
    #         state.joined_list = join(tuple_list, state.tuple_indices, state.code_list, state.code_indices)
    #     return state
    #
    # def _process_root_state(self, state, patchify):
    #     """Process root state for a single shape configuration"""
    #     unshuffled_tensor = patchify(state.tensor)
    #     state.orig_size = list(unshuffled_tensor.size())  # Update state for next iteration
    #
    #     # Determine if this is the last iteration
    #     last_shape = True if all(dim == 1 for dim in state.shape) else False
    #
    #     tensor = unshuffled_tensor[0]
    #     tuple_list = tensor_to_tuple(unshuffled_tensor, state.shape)[0]
    #     code_mapping = torch.full((tensor.size(0),), -1, dtype=torch.long)
    #     # Update code_mapping using the root vocabulary
    #     with self._lock:
    #         for i, tup in enumerate(tuple_list):
    #             code = self.inverse_vocab.get(tup, None)
    #             if code is not None:
    #                 code_mapping[i] = int(code)
    #             else:
    #                 if last_shape:
    #                     code_mapping[i] = len(self.vocab)
    #                     update_vocab(self.vocab, self.inverse_vocab, tup, len(self.vocab))
    #
    #     non_root_indices = torch.where(code_mapping == -1)[0]
    #     root_indices = torch.where(code_mapping != -1)[0]
    #     root_codes = code_mapping[root_indices].tolist()
    #     state.tensor = tensor[non_root_indices].unsqueeze(0)
    #     state.code_list.extend(list(map(str, root_codes)))
    #
    #     if len(state.tuple_indices) > 0:
    #         root_indices = torch.tensor(state.tuple_indices)[root_indices].tolist()
    #         non_root_indices = torch.tensor(state.tuple_indices)[non_root_indices].tolist()
    #     else:
    #         root_indices = root_indices.tolist()
    #         non_root_indices = non_root_indices.tolist()
    #
    #     # Update state for current shape
    #     state.tuple_indices = non_root_indices
    #     state.code_indices.extend(root_indices)
    #
    #     return state
    #
    # def _split_data(self, state, scale_factor):
    #     """Split data into tuple and code lists"""
    #     tuple_list, tuple_indices, code_list, code_indices = split(state.joined_list, scale_factor)
    #
    #     if len(tuple_list) > 0:
    #         state.orig_size[1] = len(tuple_list)  # Update the length dimension
    #         state.tensor = tuple_to_tensor(tuple_list, state.shape, state.orig_size, state.data_dtype)
    #         state.tuple_indices = tuple_indices
    #         state.code_list = code_list
    #         state.code_indices = code_indices
    #         return state
    #
    #     return None

    def _merge_pairs(self, state, min_freq, min_entrance_freq):
        while True:
            counter = FrequencyCounter(min_entrance_freq)  # TODO: why another counter here?
            # for batch_states in states.values(): # TODO: Another for loop here
            #     for state in batch_states:
            #         counter.update_merge_freq_tables(state.joined_list)
            # for state_i in states.values():
            #     counter.update_merge_freq_tables(state_i.joined_list)
            counter.update_merge_freq_tables(state.joined_list)

            freq_table = counter.get_global_freq_table()
            # print(freq_table)
            if len(freq_table) == 0:
                break

            pair, freq = counter.get_max_merge_pair().values()
            # print(pair, freq)
            if freq < min_freq:
                break
            code = self.inverse_vocab.get(pair, None)
            if code is None:
                idx = str(len(self.vocab))
                update_vocab(self.vocab, self.inverse_vocab, pair, idx)
            else:
                idx = code
            # for batch_states in states.values():
            #     for state in batch_states:
            #         state.joined_list = merge(state.joined_list, pair, idx)
                    # print(state.joined_list)
            state.joined_list = merge(state.joined_list, pair, idx)
            # TODO: need to convert
        return

    # def encode(self, data, max_shape):
    #     """
    #     Encode using the trained vocabulary.
    #
    #     Args:
    #         data (array-like): The input data that is shuffled into a list of tuples.
    #         max_shape (tuple): The initial dimension for the tuples.
    #
    #     Returns:
    #         tuple_list (list): The encoded list of tuples.
    #     """
    #
    #     if len(self.vocab) == 0:
    #         raise ValueError('Vocabulary not trained yet.')
    #
    #     tuple_list = data
    #     shapes = find_tuple_shapes(max_shape)
    #
    #     for i in range(len(shapes)):
    #         # the input is already reshaped so we skip reshaping in the first iteration
    #         if i > 0:
    #             tuple_list = tuple_reshape(tuple_list, shapes[i - 1], shapes[i])
    #
    #         # update with the root vocabulary
    #         for i, t in enumerate(tuple_list):
    #             if t in self.inverse_vocab.keys():
    #                 tuple_list[i] = self.inverse_vocab[t]
    #
    #         # merge pairs
    #         while True:
    #             stats = get_freq_pairs(tuple_list)
    #             pair, _ = get_max_pair(stats)
    #
    #             if pair not in self.inverse_vocab.keys():
    #                 break
    #
    #             # look up the pair in the inverse_vocab
    #             idx = self.inverse_vocab[pair]
    #
    #             tuple_list = merge(tuple_list, pair, idx)
    #
    #     return tuple_list
    #
    # def decode(self, encoded):
    #     """
    #     Decode the encoded data back into its original list of tuples.
    #
    #     Args:
    #         encoded (list): encoded data containing string codes.
    #
    #     Returns:
    #         plain_list (list): The decoded list of tuples.
    #     """
    #
    #     decoded = []
    #     for i in encoded:
    #         pair = self.vocab[i]
    #         decoded.append(dfs(pair, self.vocab))
    #
    #     return np.concatenate(decoded).tolist()
