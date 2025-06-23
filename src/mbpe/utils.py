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

#
# def dfs(token, vocab):
#     """
#     Recursively decode a codeword (str) into a flat tuple of integers.
#     Handles base tuples and recursive merges without sanity checks.
#
#     Args:
#         token (str or tuple): A codeword or base tuple.
#         vocab (dict): Mapping from codeword to symbol/tuple.
#
#     Returns:
#         tuple[int]: Fully flattened tuple of integers.
#     """
#     if not isinstance(token, str):
#         if isinstance(token, tuple):
#             return tuple(int(x) for x in token)
#         return (int(token),)
#
#     resolved = vocab[token]
#     return tuple(flat for part in resolved for flat in dfs(part, vocab))


# def dfs(tup, vocab):
#     """
#     Perform a depth-first search (DFS) to decode a pair recursively using the provided vocabulary.
#
#     Args:
#         tup (str): A tuple or a tuple of a pair of tuples.
#         vocab (dict): A dictionary mapping string codes to tuples representing pairs.
#
#     Returns:
#         list: A list of decoded elements resulting from the DFS decoding process.
#
#     """
#
#     while any(isinstance(item, str) for item in tup):
#         new_tup = ()
#         for i in tup:
#             if isinstance(i, str):
#                 new_tup = new_tup + vocab[i]
#             else:
#                 new_tup = new_tup + (i,)
#         tup = new_tup
#
#     return tup


def dfs(token, vocab, default=None):
    """
    Recursively decode a codeword (str or tuple) into a flat tuple of integers.

    Args:
        token (str | tuple | int): A codeword (string), or base-level tuple/int.
        vocab (dict): Mapping from codeword strings to symbol tuples (which may include more codewords).

    Returns:
        tuple[int]: Fully flattened tuple of integers.
    """
    if not isinstance(token, str):
        if isinstance(token, tuple):
            return tuple(int(x) for x in token)
        return (int(token),)

    resolved = vocab.get([token], default)
    return tuple(x for part in resolved for x in dfs(part, vocab))



def compress_indices(indices):  # TODO: can use BPE for compress indices
    if not indices:
        return []

    indices = sorted(indices)
    compressed = []
    start = prev = indices[0]

    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            compressed.append(start if start == prev else (start, prev))
            start = prev = i

    compressed.append(start if start == prev else (start, prev))
    return compressed


def decompress_indices(compressed):
    result = []
    for item in compressed:
        if isinstance(item, int):
            result.append(item)
        else:
            start, end = item
            result.extend(range(start, end + 1))
    return result
