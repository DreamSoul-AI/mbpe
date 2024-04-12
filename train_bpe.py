import json
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from collections import defaultdict

# helper functions
def freq_pair(ids):
    """
    Computes the frequency of all pairs.

    Args:
        ids (ndarray): A list of lists containing integers.

    Returns:
        pairs (defaultdict): A dictionary where keys are pairs of integers and values are their frequencies.
    """

    pairs = defaultdict(int)
    for row in ids:
        for i in range(len(row) - 1):
            pair = (int(row[i]), int(row[i+1]))
            pairs[pair] += 1
    return pairs

def max_freq_pair(pairs):
    """
    Finds the pair with the highest frequency in a dictionary of pairs and their frequencies.

    Args:
        pairs (defaultdict): A dictionary where keys are pairs of integers and values are their frequencies.

    Returns:
        best_pair (tuple): The pair with the highest frequency.
        max_freq (int): The frequency of the best pair.
    """

    max_freq = None
    for pair, freq in pairs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    return best_pair, max_freq

def merge(ids, pair, idx):
    """
    Args:
        ids (ndarray):  data to be merged.
        pair (tuple): A pair of integers to be merged.
        idx (int): The value to replace the merged pair.

    Returns:
        ndarray: Merged numpy array.
    """
    
    merged_ids = []
    for row in ids:
        merged_row = []
        i = 0
        while i < len(row):
            if i < len(row) - 1 and (row[i], row[i+1]) == pair:
                merged_row.append(idx)
                i += 2
            else:
                merged_row.append(row[i])
                i += 1
        merged_ids.append(merged_row)
    return np.array(merged_ids)


class Tokenizer:

    def __init__(self):
        self.vocab = self._create_vocab_dic()

    def get_vocab(self):
        return self.vocab
    
    def get_vocab_len(self):
        return len(self.vocab)

    def _create_vocab_dic(self):
        """
        Creates a dictionary mapping token IDs to pixel values.

        Returns:
            vocab_dic (defaultdict): A dictionary where keys are token IDs (integers) and values are pixel values (tuples).
        """
        
        vocab_dic = defaultdict(int)
        for i in range(256):
            vocab_dic[i] = i
        return vocab_dic
    
    def train_encode(self, data, vocab_size, min_freq=2):
        """
        Train and encode using the provided data.

        Args:
        - data (torch.Tensor or ndarray): Input data. Must be a 4-dimensional tensor or array.
        - vocab_size (int): Size of the vocabulary to be learned.
        - min_freq (int): Minimum frequency threshold for pairs to be included in the vocabulary. Defaults to 2.

        Returns:
        - data (ndarray): Encoded data.
        """

        if isinstance(data, torch.Tensor):
            data = data.numpy()

        shape = data.shape
        if data.ndim == 4:
            data = data.reshape(shape[0], np.prod(shape[1:])) * 255
            data = data.astype(int)
        else:
            raise ValueError('Input data must have exactly 4 dimensions.')
        
        while len(self.vocab) < vocab_size:
            pair, freq = max_freq_pair(freq_pair(data))

            if freq < min_freq:
                break

            if pair not in self.vocab.values():
                idx = len(self.vocab)
                self.vocab[idx] = pair
            else:
                for key, val in self.vocab.items():
                    if val == pair:
                        idx = key
            
            data = merge(data, pair, idx)

        self.json_save(f'./bpe_model/vocab_{len(self.vocab)}.json')
        return data

    def json_save(self, file_name):
        json_string = json.dumps(self.vocab, indent=4)
        with open(file_name, 'w') as file:
            file.write(json_string)

    def json_load(self, file_name):
        with open(file_name, 'r') as file:
            json_string = file.read()
        self.vocab = json.loads(json_string)
        
        
        
    