from collections import defaultdict

class Tokenizer:
    """Base class for Tokenizer"""

    def __init__(self):
        self.vocab = defaultdict(str)

    def get_vocab(self):
        return self.vocab
    
    def get_vocab_len(self):
        return len(self.vocab)

    # def train(self, tuple_data, vocab_size, min_freq=2):
    #     # Tokenizer can train a vocabulary of size vocab_size from given data
    #     raise NotImplementedError

    def train_encode(self, data, vocab_size, dim, min_freq):
        # Tokenizer can train and encode given data
        raise NotImplementedError

    def decode(self, encoded):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError
