import numpy as np
import torch
from collections.abc import Iterable
from collections import defaultdict, Counter
from itertools import repeat
from .tensor_shuffle import tensor_unshuffle, tensor_shuffle


def ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def tensor_to_tuple(tensor, shapes):
    reshaped_tensor = tensor.numpy().reshape(tensor.size(0), -1, np.prod(shapes))
    data = [list(map(tuple, sub_tensor)) for sub_tensor in reshaped_tensor]
    return data


def tuple_to_tensor(tuple_list, shapes, orig_size, dtype):
    reshaped_tensor = torch.tensor(tuple_list).to(dtype)
    tensor = reshaped_tensor.reshape(reshaped_tensor.size(0), -1, *shapes).reshape(orig_size)
    return tensor


def find_tuple_shapes(dim):
    """
    Find all possible shapes of tuples based on the given dimensions.

    Args:
        dim (tuple): A tuple of dimensions.

    Returns:
        shapes: A list of tuples representing all possible shapes.
    """

    # Find divisors for each dimension in desceding order
    # Each smaller divisor must also be divisible by the previous larger divisor
    # e.g. (2, 2) -> (2, 2), (1, 2), (1, 1); (6, 2) -> (6, 2), (3, 2), (1, 2), (1, 1)
    dim = list(dim)
    divisors = []
    for d in dim:
        tmp = []
        new_divisor = d
        for i in range(d - 1, 0, -1):
            if d % i == 0 and new_divisor % i == 0:
                tmp.append(i)
                new_divisor = i
        divisors.append(tmp)

    # Generate all possible shapes
    shapes = [dim]
    current_dim = dim
    for i, div in enumerate(divisors):
        for d in div:
            new_shape = current_dim.copy()
            new_shape[i] = d
            current_dim = new_shape
            shapes.append(new_shape)

    return shapes


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


def join(tuples, codes, idx):
    total_length = len(tuples) + len(codes)
    merged = np.empty(total_length, dtype=object)
    str_mask = np.zeros(total_length, dtype=bool)
    str_mask[idx] = True
    merged[str_mask] = codes
    merged[~str_mask] = np.fromiter(tuples, dtype=object)
    return list(merged)


def find_root(tensor, min_freq, vocab):
    output, inverse_indices, counts = torch.unique(tensor, return_inverse=True, return_counts=True, dim=0)
    freq_mask = counts >= min_freq
    unique_codes = torch.arange(len(vocab), len(vocab) + torch.sum(freq_mask))
    full_mapping = torch.full((counts.size(0),), -1)
    full_mapping[freq_mask] = unique_codes
    mapped_values = full_mapping[inverse_indices]
    root_indices = torch.where(mapped_values == -1)[0]
    code_indices = torch.where(mapped_values != -1)[0]
    codes = mapped_values[code_indices]
    return output[freq_mask], root_indices.tolist(), unique_codes.tolist(), codes.tolist(), code_indices.tolist()


def get_freq_pairs(mixed_list):
    """
    Computes the frequency of all pairs.

    Args:
        mixed_list (list): A list of tuples and strings

    Returns:
        counts (defaultdict): A dictionary where keys are pairs and values are their frequencies
    """

    return Counter(zip(mixed_list, mixed_list[1:]))


def get_max_pair(pairs):
    """
    Finds the pair with the highest frequency in a dictionary of pairs and their frequencies.

    Args:
        pairs (defaultdict): A dictionary where keys are pairs of tuples and values are their frequencies.

    Returns:
        max_pair (tuple): The pair with the highest frequency.
        freq (int): The frequency of the best pair.
    """

    max_pair = max(pairs, key=pairs.get)
    freq = pairs[max_pair]
    return max_pair, freq


def update_vocab(vocab, inv_vocab, pairs, indices):
    """
    Update the vocabulary and inverse vocabulary with new pair(s).

    Args:
        vocab (dict): A dictionary mapping string codes to tuples representing pairs.
        inv_vocab (dict): A dictionary mapping tuples to string codes.
        pairs (tuple or list): Either a single pair tuple or a list of pair tuples to be added.
        indices (str or list): Either a single string index or a list of indices corresponding to the pairs.

    Returns:
        None
    """

    if isinstance(pairs, tuple):
        vocab[indices] = pairs
        inv_vocab[pairs] = indices
        return

    if len(pairs) != len(indices):
        raise ValueError("Number of pairs must match number of indices")

    for pair, index in zip(pairs, indices):
        vocab[index] = pair
        inv_vocab[pair] = index

    return


def merge(tuple_list, vocab, idx):
    """
    Args:
        tuple_list (list): tuples to be merged.
        vocab (tuple): A pair of integers to be merged.
        idx (str): The value to replace the merged pair in string type.

    Returns:
        new_tuple_list: Merged tuple list.
    """

    new_tuple_list = []
    i = 0
    while i < len(tuple_list):
        if i < len(tuple_list) - 1 and (tuple_list[i], tuple_list[i + 1]) == vocab:
            new_tuple_list.append(idx)
            i += 2
        else:
            new_tuple_list.append(tuple_list[i])
            i += 1
    return new_tuple_list


def dfs(tup, vocab):
    """
    Perform a depth-first search (DFS) to decode a pair recursively using the provided vocabulary.

    Args:
        tup (tuple): A tuple representing a pair of tuples or string codes to be decoded.
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
