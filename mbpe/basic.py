from .base import BaseTokenizer
from .utils import *


class Tokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()

    def train(self, data, dim, min_freq=2, root_min_freq=2):
        """
        Train a vocabulary of size vocab_size using the provided data.

        Args:
        - data (array-like): The raw data to be used for training and encoding.
        - dim (tuple): The dimension of the tuples reshaped from the raw data.
        - min_freq (int): The minimum frequency of a pair to be considered.
        - root_min_freq (int): The minimum frequency of a pair to be considered for the root vocabulary.

        Returns:
        - None
        """

        tuple_list = reshape_to_tuples(data, dim)
        shapes = find_tuple_shapes(dim)

        for i in range(len(shapes)):
            if i > 0:
                tuple_list = reshape_tuples(tuple_list, shapes[i-1], shapes[i])

            # build root vocabulary
            if i == len(shapes) - 1:
                root_min_freq = 1
            target_tuple_list = freq_tuple(tuple_list, root_min_freq)
            root_vocab = defaultdict(str)
            for t in target_tuple_list:
                # look up the root in the inverse_vocab
                str_code = self.inverse_vocab.get(t)
                if str_code:
                    idx = str_code
                else:
                    idx = str(len(self.vocab))
                    update_vocab(self.vocab, self.inverse_vocab, t, idx)
                root_vocab[t] = idx

            # update the list of tuples with the root vocabulary
            tuple_list = [root_vocab[t]
                          if t in root_vocab else t for t in tuple_list]

            while True:
                stats = get_freq(tuple_list)
                pair, freq = get_max_pair(stats)

                if freq < min_freq:
                    break

                # look up the pair in the inverse_vocab
                str_code = self.inverse_vocab.get(pair)
                if str_code:
                    idx = str_code
                else:
                    idx = str(len(self.vocab))
                    update_vocab(self.vocab, self.inverse_vocab, pair, idx)

                tuple_list = merge(tuple_list, pair, idx)

        return

    def train_encode(self, data, dim, min_freq=2, root_min_freq=2):
        """
        Train and encode using the provided data.

        Args:
        - data (array-like): The raw data to be used for training and encoding.
        - dim (tuple): The dimension of the tuples reshaped from the raw data.
        - min_freq (int): The minimum frequency of a pair to be considered.
        - root_min_freq (int): The minimum frequency of a pair to be considered for the root vocabulary.

        Returns:
        - tuple_list (list): The encoded list of tuples.
        """

        tuple_list = reshape_to_tuples(data, dim)
        shapes = find_tuple_shapes(dim)

        for i in range(len(shapes)):
            if i > 0:
                tuple_list = reshape_tuples(tuple_list, shapes[i-1], shapes[i])

            # build root vocabulary
            if i == len(shapes) - 1:
                root_min_freq = 1
            target_tuple_list = freq_tuple(tuple_list, root_min_freq)
            root_vocab = defaultdict(str)
            for t in target_tuple_list:
                # look up the root in the inverse_vocab
                str_code = self.inverse_vocab.get(t)
                if str_code:
                    idx = str_code
                else:
                    idx = str(len(self.vocab))
                    update_vocab(self.vocab, self.inverse_vocab, t, idx)
                root_vocab[t] = idx

            # update the list of tuples with the root vocabulary
            tuple_list = [root_vocab[t]
                          if t in root_vocab else t for t in tuple_list]

            while True:
                stats = get_freq(tuple_list)
                pair, freq = get_max_pair(stats)

                if freq < min_freq:
                    break

                # look up the pair in the inverse_vocab
                str_code = self.inverse_vocab.get(pair)
                if str_code:
                    idx = str_code
                else:
                    idx = str(len(self.vocab))
                    update_vocab(self.vocab, self.inverse_vocab, pair, idx)

                tuple_list = merge(tuple_list, pair, idx)

        # if self.decode(tuple_list) != plain_list:
        #     raise ValueError('Encoding Decoding Mismatch')

        return tuple_list

    def encode(self, data, dim, min_freq=2, root_min_freq=2):
        """
        Encode using the trained vocabulary.

        Args:
        - data (array-like): The raw data to be used for training and encoding.
        - dim (tuple): The dimension of the tuples reshaped from the raw data.
        - min_freq (int): The minimum frequency of a pair to be considered.
        - root_min_freq (int): The minimum frequency of a pair to be considered for the root vocabulary.

        Returns:
        - tuple_list (list): The encoded list of tuples.
        """

        if len(self.vocab) == 0:
            raise ValueError('Vocabulary not trained yet.')

        tuple_list = reshape_to_tuples(data, dim)
        shapes = find_tuple_shapes(dim)

        for i in range(len(shapes)):
            if i > 0:
                tuple_list = reshape_tuples(tuple_list, shapes[i-1], shapes[i])

            # build root vocabulary
            if i == len(shapes) - 1:
                root_min_freq = 1
            target_tuple_list = freq_tuple(tuple_list, root_min_freq)
            root_vocab = defaultdict(str)
            for t in target_tuple_list:
                # look up the root in the inverse_vocab
                str_code = self.inverse_vocab.get(t)
                idx = str_code
                root_vocab[t] = idx

            # update the list of tuples with the root vocabulary
            tuple_list = [root_vocab[t]
                          if t in root_vocab else t for t in tuple_list]

            while True:
                stats = get_freq(tuple_list)
                pair, freq = get_max_pair(stats)

                if freq < min_freq:
                    break

                # look up the pair in the inverse_vocab
                str_code = self.inverse_vocab.get(pair)
                idx = str_code

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
