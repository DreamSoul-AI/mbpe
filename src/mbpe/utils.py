import numpy as np
import torch
from collections.abc import Iterable
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
