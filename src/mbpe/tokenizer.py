import torch
from .utils import *
from collections import defaultdict


class BaseTokenizer:
    """Base class for Tokenizer"""

    def __init__(self):
        self.vocab = defaultdict(tuple)
        self.inverse_vocab = defaultdict(int)
        self.vocab[''] = 0
        self.inverse_vocab[0] = ''

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

        dtype = data.dtype
        orig_size = data.size()
        shapes = find_tuple_shapes(max_shape)

        tuple_indices = []
        code_list = []
        code_indices = []
        for shape in shapes:
            n = len(orig_size)
            dim_index = list(range(n - 3, n))   # only last 3 dimensions are considered
            # print(dim_index)
            _tuple = ntuple(len(dim_index))
            shape = list(_tuple(shape))

            scale_factor = [orig_size[dim_index[i]] // shape[i] for i in range(len(dim_index))]
            # 1. split the data into tuples and codes (exclude the first iteration)
            if len(code_list) > 0:
                data, tuple_indices, code_list, code_indices = split(data, list(set(scale_factor))[-1])
                orig_size[1] = len(data)    # update the length dimension
                data = tuple_to_tensor(data, shape, orig_size, dtype)
            # 2. reshape the tensor
            unshuffled_tensor = tensor_unshuffle(data, scale_factor, dim_index)
            orig_size = list(unshuffled_tensor.size())    # store for next iteration
            # print(orig_size)
            
            # 3. find root vocabulary
            if shape == [1, 1, 1]:
                root_min_freq = 1
            root, root_indices, codes, indices = find_root(unshuffled_tensor, root_min_freq, self.vocab)
            update_vocab(self.vocab, self.inverse_vocab, tensor_to_tuple(root, shape)[0], sorted(set(codes)))

            tuple_list = tensor_to_tuple(unshuffled_tensor[:, root_indices], shape)[0]  #TODO: why [0]?
            if len(tuple_indices) > 0:
                indices = torch.tensor(tuple_indices)[indices].tolist()
            code_list.extend(codes)
            code_indices.extend(indices)
            data = join(tuple_list, code_list, code_indices)
            # print(data)
            
            # 4. merge
            while True:
                stats = get_freq_pairs(data)
                pair, freq = get_max_pair(stats)

                if freq < min_freq:
                    break

                # look up the pair in the inverse_vocab
                code = self.inverse_vocab[pair]
                if code != 0:
                    idx = code
                else:
                    idx = len(self.vocab)
                    update_vocab(self.vocab, self.inverse_vocab, pair, idx)

                data = merge(data, pair, idx)
            
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
