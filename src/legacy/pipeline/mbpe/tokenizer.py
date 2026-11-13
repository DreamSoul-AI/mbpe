from .utils import *
from collections import defaultdict


class BaseTokenizer:
    """Base class for Tokenizer"""

    def __init__(self):
        self.vocab = defaultdict(tuple)
        self.vocab['0'] = '[PAD]'
        self.inverse_vocab = defaultdict(str)

    def get_vocab(self):
        return self.vocab

    def get_vocab_len(self):
        return len(self.vocab)

    def train(self, data, dim, min_freq, root_min_freq):
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

    def train(self, data, dim, min_freq=2, root_min_freq=2):
        """
        Train a vocabulary of size vocab_size using the provided data.

        Args:
        - data (array-like): A list of tuples reshaped from the raw image.
        - dim (tuple): The initial dimension of each tuple.
        - min_freq (int): The minimum frequency of a pair to be considered.
        - root_min_freq (int): The minimum frequency of a pair to be considered for the root vocabulary.

        Returns:
        - None
        """

        tuple_list = data
        shapes = find_tuple_shapes(dim)

        for i in range(len(shapes)):
            # the input is already reshaped so we skip reshaping in the first iteration
            if i > 0:
                tuple_list = reshape_tuples(tuple_list, shapes[i-1], shapes[i])

            # build root vocabulary
            if i == len(shapes) - 1:
                # set root_min_freq to 1 for the last iteration where each tuple only includes 1 element
                root_min_freq = 1
            target_tuple_list = get_freq_tuples(tuple_list, root_min_freq)
            root_vocab = defaultdict(str)
            for t in target_tuple_list:
                # look up the root in the inverse_vocab
                str_code = self.inverse_vocab[t]
                if str_code != '':
                    idx = str_code
                else:
                    idx = str(len(self.vocab))
                    update_vocab(self.vocab, self.inverse_vocab, t, idx)
                root_vocab[t] = idx

            # update the list of tuples with the root vocabulary
            tuple_list = [root_vocab[t]
                          if t in root_vocab else t for t in tuple_list]

            while True:
                stats = get_freq_pairs(tuple_list)
                pair, freq = get_max_pair(stats)

                if freq < min_freq:
                    break

                # look up the pair in the inverse_vocab
                str_code = self.inverse_vocab[pair]
                if str_code != '':
                    idx = str_code
                else:
                    idx = str(len(self.vocab))
                    update_vocab(self.vocab, self.inverse_vocab, pair, idx)

                tuple_list = merge(tuple_list, pair, idx)

        return

    def encode(self, data, dim):
        """
        Encode using the trained vocabulary.

        Args:
        - data (array-like): A list of tuples reshaped from the raw image.
        - dim (tuple): The initial dimension of each tuple.

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
                tuple_list = reshape_tuples(tuple_list, shapes[i-1], shapes[i])
            
            # update with the root vocabulary
            for i, t in enumerate(tuple_list):
                if t in self.inverse_vocab.keys():
                    tuple_list[i] = self.inverse_vocab[t]
            
            # merge pairs
            while True:
                stats = get_freq_pairs(tuple_list)
                pair, freq = get_max_pair(stats)

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
