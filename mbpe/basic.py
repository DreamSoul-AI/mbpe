from .base import BaseTokenizer
from .utils import *


class Tokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()

    def train(self, data, dim, min_freq=2):
        """
        Train a vocabulary of size vocab_size using the provided data.

        Args:
        - data(array-like): The raw data to be used for training and encoding.
        - dim(tuple): The dimension of the tuples reshaped from the raw data.
        - min_freq(int): The minimum frequency of a pair to be considered.

        Returns:
        - None
        """

        shapes = find_tuple_shapes(dim)
        tuple_list = data

        for i in range(len(shapes)):
            if i == 0:
                tuple_list = reshape_to_tuples(tuple_list, shapes[i])
            else:
                tuple_list = reshape_tuples(tuple_list, shapes[i-1], shapes[i])

            # build root vocabulary
            tmp_vocab = defaultdict(str)
            if i == len(shapes) - 1:
                root_min_freq = 1
            else:
                root_min_freq = min_freq
            target_tuple_list = freq_tuple(tuple_list, root_min_freq)
            for t in target_tuple_list:
                if t not in self.vocab.values():
                    idx = str(len(self.vocab))
                    self.vocab[idx] = t
                else:
                    for vocab_k, vocab_v in self.vocab.items():
                        if vocab_v == t:
                            idx = vocab_k
                tmp_vocab[t] = idx
            self.inverse_vocab.update(tmp_vocab)
            tuple_list = [tmp_vocab[t]
                          if t in tmp_vocab else t for t in tuple_list]

            while True:
                pair, freq = max_freq_pair(freq_pair(tuple_list))

                if freq < min_freq:
                    break

                if pair not in self.vocab.values():
                    idx = str(len(self.vocab))
                    self.vocab[idx] = pair
                    self.inverse_vocab[pair] = idx
                else:
                    for key, val in self.vocab.items():
                        if val == pair:
                            idx = key
                tuple_list = merge(tuple_list, pair, idx)

        return

    def train_encode(self, data, dim, min_freq=2):
        """
        Train and encode using the provided data.

        Args:
        - data(array-like): The raw data to be used for training and encoding.
        - vocab_size(int): The size of the vocabulary
        - dim(tuple): The dimension of the tuples reshaped from the raw data.
        - min_freq(int): The minimum frequency of a pair to be considered.

        Returns:
        - tuple_list(list): The encoded list of tuples.
        """

        shapes = find_tuple_shapes(dim)
        tuple_list = data

        for i in range(len(shapes)):
            if i == 0:
                tuple_list = reshape_to_tuples(tuple_list, shapes[i])
            else:
                tuple_list = reshape_tuples(tuple_list, shapes[i-1], shapes[i])

            # build root vocabulary
            tmp_vocab = defaultdict(str)
            if i == len(shapes) - 1:
                root_min_freq = 1
            else:
                root_min_freq = min_freq
            target_tuple_list = freq_tuple(tuple_list, root_min_freq)
            for t in target_tuple_list:
                if t not in self.vocab.values():
                    idx = str(len(self.vocab))
                    self.vocab[idx] = t
                else:
                    for vocab_k, vocab_v in self.vocab.items():
                        if vocab_v == t:
                            idx = vocab_k
                tmp_vocab[t] = idx
            self.inverse_vocab.update(tmp_vocab)
            tuple_list = [tmp_vocab[t]
                          if t in tmp_vocab else t for t in tuple_list]

            while True:
                pair, freq = max_freq_pair(freq_pair(tuple_list))

                if freq < min_freq:
                    break

                if pair not in self.vocab.values():
                    idx = str(len(self.vocab))
                    self.vocab[idx] = pair
                    self.inverse_vocab[pair] = idx
                else:
                    for key, val in self.vocab.items():
                        if val == pair:
                            idx = key
                tuple_list = merge(tuple_list, pair, idx)

        # if self.decode(tuple_list) != tuple_list:
        #     print(tuple_list)
        #     print(tuple_list)
        #     raise ValueError('Encoding Decoding Mismatch')

        return tuple_list

    def encode(self, data, dim, min_freq=2):
        """
        Encode using the trained vocabulary.

        Args:
        - data(array-like): The raw data to be used for training and encoding.
        - dim(tuple): The dimension of the tuples reshaped from the raw data.
        - min_freq(int): The minimum frequency of a pair to be considered.

        Returns:
        - tuple_list(list): The encoded list of tuples.
        """

        if len(self.vocab) == 0:
            raise ValueError('Vocabulary not trained yet.')

        shapes = find_tuple_shapes(dim)
        tuple_list = data

        for i in range(len(shapes)):
            if i == 0:
                tuple_list = reshape_to_tuples(tuple_list, shapes[i])
            else:
                tuple_list = reshape_tuples(tuple_list, shapes[i-1], shapes[i])

            # build root vocabulary
            tmp_vocab = defaultdict(str)
            if i == len(shapes) - 1:
                root_min_freq = 1
            else:
                root_min_freq = min_freq
            target_tuple_list = freq_tuple(tuple_list, root_min_freq)
            for t in target_tuple_list:
                if t not in self.vocab.values():
                    idx = str(len(self.vocab))
                    self.vocab[idx] = t
                else:
                    for vocab_k, vocab_v in self.vocab.items():
                        if vocab_v == t:
                            idx = vocab_k
                tmp_vocab[t] = idx
            tuple_list = [tmp_vocab[t]
                          if t in tmp_vocab else t for t in tuple_list]

            while True:
                pair, freq = max_freq_pair(freq_pair(tuple_list))

                if freq < min_freq:
                    break

                for key, val in self.vocab.items():
                    if val == pair:
                        idx = key
                tuple_list = merge(tuple_list, pair, idx)

        # if self.decode(tuple_list) != tuple_list:
        #     print(tuple_list)
        #     print(tuple_list)
        #     raise ValueError('Encoding Decoding Mismatch')

        return tuple_list

    def decode(self, encoded):
        """
        Decode the encoded data back into its original list of tuples.

        Args:
        - encoded(list): encoded data containing tuples and string codes.

        Returns:
        - decoded(list): The decoded list of tuples.
        """

        decoded = []
        for i in encoded:
            if isinstance(i, str):
                pair = self.vocab[i]
                decoded = decoded + dfs(pair, self.vocab)
            else:
                decoded.append(i)
        return decoded
