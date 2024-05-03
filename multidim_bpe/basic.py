from .base import Tokenizer
from .utils import *

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train_encode(self, data, vocab_size, dim=(2, 2), min_freq=2):
        """
        Train and encode using the provided data.

        Args:
        - data: raw data to be trained and encoded.
        - vocab_size(int): The size of the vocabulary
        - min_freq(int): The minimum frequency of a pair to be considered.

        Returns:
        - tuple_list(numpy.ndarray): The encoded list of tuples.
        """

        tuple_list = reshape_to_tuples(data, dim)
        
        new_tuple_list = tuple_list
        while len(self.vocab) < vocab_size:
            pair, freq = max_freq_pair(freq_pair(new_tuple_list))

            if freq < min_freq:
                break
            
            if pair not in self.vocab.values():
                idx = str(len(self.vocab))
                self.vocab[idx] = pair
            else:
                for key, val in self.vocab.items():
                    if val == pair:
                        idx = key      
            new_tuple_list = merge(new_tuple_list, pair, idx)
        else:
            _, freq = max_freq_pair(freq_pair(new_tuple_list))
            while freq >= min_freq:
                pair, freq = max_freq_pair(freq_pair(new_tuple_list))

                if pair not in self.vocab.values():
                    break
                
                for key, val in self.vocab.items():
                    if val == pair:
                        idx = key
                        new_tuple_list = merge(new_tuple_list, pair, idx)

        if self.decode(new_tuple_list) != tuple_list:
            print(new_tuple_list)
            print(tuple_list)
            raise ValueError('Encoding Decoding Mismatch')

        return new_tuple_list

    def decode(self, encoded):
        decoded = []
        for i in encoded:
            if isinstance(i, str):
                pair = self.vocab[i]
                decoded = decoded + dfs(pair, self.vocab)
            else:
                decoded.append(i)
        return decoded