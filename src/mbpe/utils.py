import torch
import numpy as np
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


def tensor_to_tuples(tensor, dim, dim_index):
    """
    Reshape the given data into tuples based on the specified dimensions.

    Args:
        data (tensor): The input image to be reshaped into tuples.
        dim (tuple): A 2-tuple specifying the desired dimensions of the tuples.

    Returns:
        tuples: A list of tuples generated through reshaping the data.

    Raises:
        ValueError: If the input data has more than 3 dimensions.
    """
    _tuple = ntuple(len(dim_index))
    size = tensor.size()
    dim = _tuple(dim)
    scale_factor = [size[dim_index[i]] // dim[i] for i in range(len(dim_index))]
    unshuffled_tensor = tensor_unshuffle(tensor, scale_factor, dim_index)
    shuffled_tensor = tensor_shuffle(unshuffled_tensor, scale_factor, dim_index)
    print(unshuffled_tensor.size())
    print(shuffled_tensor.size())
    tuples = list(map(tuple, unshuffled_tensor.numpy().reshape(-1, np.prod(dim))))
    return tuples


def find_tuple_shapes(dim):
    num1, num2 = dim[0], dim[1]

    divisor1 = []
    for i in range(num1 - 1, 0, -1):
        if num1 % i == 0:
            divisor1.append(i)

    divisor2 = []
    for i in range(num2 - 1, 0, -1):
        if num2 % i == 0:
            divisor2.append(i)

    tuple_shapes = [dim]
    for div in divisor1:
        tuple_shapes.append((div, num2))
    for div in divisor2:
        tuple_shapes.append((1, div))

    return tuple_shapes


def reshape_tuples(data, reshape_from, reshape_to):
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

    row_offsets = reshape_from[0] // reshape_to[0]
    col_offsets = reshape_from[1] // reshape_to[1]

    # new = []
    # for i in data:
    #     if isinstance(i, tuple):
    #         np_array = np.array(i).reshape(reshape_from)
    #         row_indices = np.arange(reshape_from[0]).reshape(
    #             row_offsets, reshape_to[0], order='F')
    #         col_indices = np.arange(reshape_from[1]).reshape(
    #             col_offsets, reshape_to[1], order='F')

    #         tuples = [
    #             tuple(
    #                 np_array[np.ix_(row_indices_group, col_indices_group)].flatten())
    #             for row_indices_group in row_indices
    #             for col_indices_group in col_indices
    #         ]
    #         new += tuples
    #     else:
    #         new.append(i)

    # print(new)

    offsets = col_offsets if row_offsets == 1 else row_offsets

    # separate tuples and strings
    tuples = []
    strings = []
    idx = []
    for i in data:
        if isinstance(i, tuple):
            tuples.append(i)
        else:
            strings.append(i)
            idx.append(len(tuples) * offsets + len(strings) - 1)

    # tuples to numpy array
    ori_shape = (len(tuples),) + reshape_from
    ori_arr = np.array(tuples).reshape(ori_shape)

    new_shape = (offsets * len(tuples),) + reshape_to
    # reshape based on the dimension that changes
    if row_offsets == 1:  # i.e. (n, 2, 2) -> (2n, 2, 1)
        new_arr = ori_arr.reshape(ori_shape[0], np.prod(
            reshape_from) // offsets, offsets).transpose(0, 2, 1)
    elif col_offsets == 1:  # i.e. (n, 2, 2) -> (2n, 1, 2)
        new_arr = ori_arr.reshape(new_shape).transpose(1, 0, 2)

    # numpy array back to tuples
    tuples = list(map(tuple, new_arr.reshape(
        new_shape[0], np.prod(reshape_to))))

    # merge tuples and strings back to one list
    total_length = len(tuples) + len(strings)
    merged = np.empty(total_length, dtype=object)
    str_mask = np.zeros(total_length, dtype=bool)
    str_mask[idx] = True
    merged[str_mask] = strings
    merged[~str_mask] = np.fromiter(tuples, dtype=object)

    return list(merged)


def freq_tuple(tuple_list, min_freq):
    """
    Select tuple elements that appear more than min_freq times

    Args:
        tuple_list (list): A list of tuples
        min_freq (int): The minimum frequency of a pair to be considered.

    Returns:
        filtered_array (list): A filtered list of tuples.
    """

    new_tuple_list = [item for item in tuple_list if isinstance(item, tuple)]
    np_tuple_list = np.fromiter(new_tuple_list, dtype=object)
    unique_elements, counts = np.unique(np_tuple_list, return_counts=True)
    return unique_elements[counts >= min_freq]


def get_freq(tuple_list):
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
