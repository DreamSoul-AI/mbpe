import numpy as np
import torch
from collections.abc import Iterable
from collections import defaultdict
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


# def unshuffle_tensor(tensor, dim, dim_index):
#     """
#     Reshape the given data into tuples based on the specified dimensions.
#
#     Args:
#         data (tensor): The input image to be reshaped into tuples.
#         dim (tuple): A 2-tuple specifying the desired dimensions of the tuples.
#
#     Returns:
#         tuples: A list of tuples generated through reshaping the data.
#
#     Raises:
#         ValueError: If the input data has more than 3 dimensions.
#     """
#     # _tuple = ntuple(len(dim_index))
#     size = tensor.size()
#     # dim = list(_tuple(dim))
#     scale_factor = [size[dim_index[i]] // dim[i] for i in range(len(dim_index))]
#     unshuffled_tensor = tensor_unshuffle(tensor, scale_factor, dim_index)
#     # tuples = list(map(tuple, unshuffled_tensor.numpy().reshape(-1, np.prod(dim))))
#     return unshuffled_tensor


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
            new_shape = list(current_dim)
            new_shape[i] = d
            current_dim = tuple(new_shape)
            shapes.append(tuple(new_shape))

    return shapes


def split(data, scale_factor):
    tuples = []
    strings = []
    idx = []
    for i in data:
        if isinstance(i, tuple):
            tuples.append(i)
        else:
            idx.append(len(tuples) * scale_factor + len(strings))
            strings.append(i)
    return tuples, strings, idx


def join(tuples, strings, idx):
    total_length = len(tuples) + len(strings)
    merged = np.empty(total_length, dtype=object)
    str_mask = np.zeros(total_length, dtype=bool)
    str_mask[idx] = True
    merged[str_mask] = strings
    merged[~str_mask] = np.fromiter(tuples, dtype=object)
    return list(merged)


def tuple_reshape(data, reshape_from, reshape_to):
    """
    Reshape the given data into tuples based on the specified dimensions.

    Args:
        data (list): A list of combination of tuples and strings.
        reshape_from (tuple): A 2-tuple specifying the current dimensions of the tuples.
        reshape_to (tuple): A 2-tuple specifying the desired dimensions of the tuples.

    Returns:
        tuples: A list of tuples generated through reshaping the data.

    Raises:
        ValueError: If the input data has more than 3 dimensions.
    """

    # find the dimension that changes
    downscale_factor = None
    downscale_factor_idx = None
    base_index = []
    index_accum = 0
    for i in range(len(reshape_from)):
        base_index.append(i + index_accum + 1)
        if reshape_from[i] != reshape_to[i]:
            downscale_factor = reshape_from[i] // reshape_to[i]
            downscale_factor_idx = i + 1 + 1
            index_accum += 1

            # separate tuples and strings
    tuples, strings, str_idx = split(data, downscale_factor)

    original_shape = (len(tuples),) + reshape_from
    original_array = np.array(tuples).reshape(original_shape)
    new_shape = [len(tuples)] + list(reshape_to)
    new_shape.insert(downscale_factor_idx, downscale_factor)
    transpose_order = [0, downscale_factor_idx] + base_index
    new_array = original_array.reshape(new_shape).transpose(transpose_order)
    new_len = len(tuples) * downscale_factor
    downscaled_tuples = list(map(tuple, new_array.reshape(new_len, np.prod(reshape_to))))

    # merge tuples and strings back
    reshaped = join(downscaled_tuples, strings, str_idx)

    return reshaped


def find_root_indices(tensor, min_freq, vocab):
    _, inverse_indices, counts = torch.unique(tensor, return_inverse=True, return_counts=True, dim=1)
    count_indices = torch.nonzero(counts >= min_freq).squeeze()
    code = torch.arange(len(vocab), len(vocab) + count_indices.size(0))
    full_mapping = torch.full((counts.size(0),), -1)
    full_mapping[count_indices] = code
    mapped_values = full_mapping[inverse_indices]
    tuple_indices = torch.where(mapped_values == -1)[0]
    code_indices = torch.where(mapped_values != -1)[0]
    print(code_indices)
    exit()
    return tuple_indices, code_indices


def filter_data(data, min_freq):
    """
    Select tuple elements that appear more than min_freq times

    Args:
        tuple_list (list): A list of tuples
        min_freq (int): The minimum frequency of a pair to be considered.

    Returns:
        filtered_array (list): A filtered list of tuples.
    """

    new_tuple_list = [item for item in data if isinstance(item, tuple)]
    np_tuple_list = np.fromiter(new_tuple_list, dtype=object)
    unique_elements, counts = np.unique(np_tuple_list, return_counts=True)
    unique_elements = unique_elements[counts >= min_freq]
    return unique_elements


def get_freq_pairs(tuple_list):  # TODO: try without for loop
    """
    Computes the frequency of all tuple pairs.

    Args:
        tuple_list (list): A list of tuples

    Returns:
        counts (defaultdict): A dictionary where keys are pairs of tuples and values are their frequencies
    """

    counts = defaultdict(int)
    for i in range(len(tuple_list) - 1):
        pair = (tuple_list[i], tuple_list[i + 1])
        counts[pair] += 1
    return counts


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


def update_vocab(vocab, inv_vocab, pair, idx):
    """
    Update the vocabulary and inverse vocabulary with a new pair.

    Args:
        vocab (dict): A dictionary mapping string codes to tuples representing pairs.
        inv_vocab (dict): A dictionary mapping tuples to string codes.
        pair (tuple): A pair of integers to be added to the vocabulary.
        idx (str): The value to replace the merged pair in string type.

    Returns:

    """

    vocab[idx] = pair
    inv_vocab[pair] = idx


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
