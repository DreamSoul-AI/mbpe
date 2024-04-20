import json
import numpy as np
import torch

from collections import defaultdict

# helper functions
def reshape_to_tuples(data, dim):
    if isinstance(data, torch.Tensor):
        if data.shape[0] == 1:
            data = data.squeeze().numpy()
    
    rows, cols = data.shape
    row_group_size = rows // dim[0]
    col_group_size = cols // dim[1]
    
    row_indices = []
    col_indices = []
    for i in range(0, row_group_size):
        row_indices.append([j for j in range(i, rows, row_group_size)])
        
    for i in range(0, col_group_size):
        col_indices.append([j for j in range(i, cols, col_group_size)])

    tuples = []

    for row_indices_group in row_indices:
        for col_indices_group in col_indices:
            group = []
            indices_tuple = []

            for r_idx in row_indices_group:
                for c_idx in col_indices_group:
                    group.append(data[r_idx, c_idx])
            
            tuples.append(tuple(group))
            
    return tuples

def freq_pair(tuple_list):
    """
    Computes the frequency of all tuple pairs.

    Args:
        tuple_list (list): A list of tuples

    Returns:
        pairs (defaultdict): A dictionary where keys are pairs of tuples and values are their frequencies
    """

    pairs = defaultdict(int)
    for i in range(len(tuple_list) - 1):
        pair = (tuple_list[i], tuple_list[i+1])
        pairs[pair] += 1
    return pairs

def max_freq_pair(pairs):
    """
    Finds the pair with the highest frequency in a dictionary of pairs and their frequencies.

    Args:
        pairs (defaultdict): A dictionary where keys are pairs of tuples and values are their frequencies.

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

def merge(tuple_list, pair, idx):
    """
    Args:
        tuple_list (list):  tuples to be merged.
        pair (tuple): A pair of integers to be merged.
        idx (int): The value to replace the merged pair.

    Returns:
        new_tuple_list: Merged tuple list.
    """
    
    new_tuple_list = []
    i = 0
    while i < len(tuple_list):
        if i < len(tuple_list) - 1 and (tuple_list[i], tuple_list[i+1]) == pair:
            new_tuple_list.append(idx)
            i += 2
        else:
            new_tuple_list.append(tuple_list[i])
            i += 1
    return new_tuple_list


class Tokenizer:

    def __init__(self):
        self.vocab = defaultdict(str)

    def get_vocab(self):
        return self.vocab
    
    def get_vocab_len(self):
        return len(self.vocab)
    
    def train_encode(self, tokens_tuple_list, vocab_size, min_freq=2):
        """
        Train and encode using the provided data.

        Args:
        - 

        Returns:
        - 
        """

        while len(self.vocab) < vocab_size:
            pair, freq = max_freq_pair(freq_pair(tokens_tuple_list))

            if freq < min_freq:
                break
            
            if pair not in self.vocab.values():
                idx = str(len(self.vocab))
                self.vocab[idx] = pair
            else:
                for key, val in self.vocab.items():
                    if val == pair:
                        idx = key
                        
            tokens_tuple_list = merge(tokens_tuple_list, pair, idx)

        # self.json_save(f'./bpe_model/vocab_{len(self.vocab)}.json')
        return tokens_tuple_list

    def json_save(self, file_name):
        new_vocab = self.vocab.copy()
        for key, val in new_vocab.items():
            l = []
            for tu in list(val):
                if isinstance(tu, tuple):
                    ll = []
                    for i in tu:
                        ll.append(int(i))
                    l.append(tuple(ll))
            if len(l) > 0:
                new_vocab[key] = tuple(l)

        json_string = json.dumps(new_vocab, indent=4)
        with open(file_name, 'w') as file:
            file.write(json_string)

    def json_load(self, file_name):
        with open(file_name, 'r') as file:
            json_string = file.read()
        self.vocab = json.loads(json_string)