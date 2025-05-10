import numpy as np
import torch
from collections.abc import Iterable
from collections import defaultdict
from itertools import repeat


def ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


class AddSequenceDim:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        return x.unsqueeze(self.dim)


def tensor_to_tuple(tensor, shapes, is_batch=True):
    if is_batch:
        reshaped_tensor = tensor.numpy().reshape(tensor.size(0), -1, np.prod(shapes))
        tuples = [list(map(tuple, sub_tensor)) for sub_tensor in reshaped_tensor]
    else:
        reshaped_tensor = tensor.numpy().reshape(-1, np.prod(shapes))
        tuples = list(map(tuple, reshaped_tensor))
    return tuples


def tuple_to_tensor(tuples, shapes, dtype, is_batch=True):
    reshaped_tensor = torch.tensor(tuples).to(dtype)
    if is_batch:
        tensor = reshaped_tensor.reshape(reshaped_tensor.size(0), -1, *shapes[1:])
    else:
        tensor = reshaped_tensor.reshape(-1, *shapes[1:])
    return tensor


def split(data, scale_factor):
    tuples = []
    tuples_indices = []
    codes = []
    code_indices = []

    for item in data:
        base_offset = len(tuples) * scale_factor + len(codes)
        if isinstance(item, tuple):
            tuples.append(item)
            tuples_indices.extend(range(base_offset, base_offset + scale_factor))
        else:
            codes.append(item)
            code_indices.append(base_offset)

    return tuples, tuples_indices, codes, code_indices

# def update_vocab(vocab, inv_vocab, msg, code):
#     if isinstance(msg, tuple):
#         vocab[code] = msg
#         inv_vocab[msg] = code
#     elif isinstance(msg, list) and isinstance(msg, list):
#         if len(msg) != len(code):
#             raise ValueError("Number of pairs must match number of indices")
#         for msg_i, code_i in zip(msg, code):
#             vocab[code_i] = msg_i
#             inv_vocab[msg_i] = code_i
#     else:
#         raise ValueError('Not valid msg and code')
#     return


# def get_freq_pairs(mixed_list, freq_table=None, n=None):
#     """
#     Computes the frequency of all pairs.
#
#     Args:
#         mixed_list (list): A list of tuples and strings
#
#     Returns:
#         counts (defaultdict): A dictionary where keys are pairs and values are their frequencies
#     """
#
#     counts = defaultdict(int)
#     for i in range(len(mixed_list) - 1):
#         pair = (mixed_list[i], mixed_list[i + 1])
#         counts[pair] += 1
#     if freq_table is not None:
#         if n is None:
#             raise ValueError("n must be provided if freq_table is not None")
#         for key, value in counts.items():
#             if key in freq_table:
#                 freq_table[key] += value
#             elif value >= n:
#                 freq_table[key] = value
#         return freq_table
#     return counts


# def get_max_pair(pairs):
#     """
#     Finds the pair with the highest frequency in a dictionary of pairs and their frequencies.
#
#     Args:
#         pairs (defaultdict): A dictionary where keys are pairs of tuples and values are their frequencies.
#
#     Returns:
#         max_pair (tuple): The pair with the highest frequency.
#         freq (int): The frequency of the best pair.
#     """
#
#     max_pair = max(pairs, key=pairs.get)
#     freq = pairs[max_pair]
#     return max_pair, freq

def dfs(tup, vocab):
    """
    Perform a depth-first search (DFS) to decode a pair recursively using the provided vocabulary.

    Args:
        tup (str): A tuple or a tuple of a pair of tuples.
        vocab (dict): A dictionary mapping string codes to tuples representing pairs.

    Returns:
        list: A list of decoded elements resulting from the DFS decoding process.

    """

    while any(isinstance(item, str) for item in tup):
        new_tup = ()
        for i in tup:
            if isinstance(i, str):
                new_tup = new_tup + vocab[i]
            else:
                new_tup = new_tup + (i,)
        tup = new_tup

    return tup
