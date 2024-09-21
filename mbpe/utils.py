import numpy as np

from collections import defaultdict

# helper functions


def find_tuple_shapes(dim):
    num1, num2 = dim[0], dim[1]

    divisor1 = []
    for i in range(num1-1, 0, -1):
        if num1 % i == 0:
            divisor1.append(i)

    divisor2 = []
    for i in range(num2-1, 0, -1):
        if num2 % i == 0:
            divisor2.append(i)

    tuple_shapes = [dim]
    for div in divisor1:
        tuple_shapes.append((div, num2))
    for div in divisor2:
        tuple_shapes.append((1, div))

    return tuple_shapes


def reshape_to_tuples(data, dim):
    """
    Reshape the given data into tuples based on the specified dimensions.

    Args:
        data (array-like): The input data to be reshaped into tuples.
        dim (tuple): A 2-tuple specifying the desired dimensions of the tuples.

    Returns:
        tuples: A list of tuples generated through reshaping the data.

    Raises:
        ValueError: If the input data has more than 3 dimensions.
    """

    if len(data.shape) == 2:
        num_rows, num_cols = data.shape
    else:
        raise ValueError("Havn't implemented for 3D data yet")

    row_offsets = num_rows // dim[0]
    col_offsets = num_cols // dim[1]

    row_indices = np.arange(num_rows).reshape(row_offsets, dim[0], order='F')
    col_indices = np.arange(num_cols).reshape(col_offsets, dim[1], order='F')

    tuples = [
        tuple(data[np.ix_(row_indices_group, col_indices_group)].flatten())
        for row_indices_group in row_indices
        for col_indices_group in col_indices
    ]

    return tuples


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

    if row_offsets == 1:
        transpose_axes = (0, 2, 1)
        offsets = col_offsets
    elif col_offsets == 1:
        transpose_axes = (0, 1, 2)
        offsets = row_offsets

    tuples = []
    strings = []
    idx = []
    # new = []
    for i in data:
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

        if isinstance(i, tuple):
            tuples.append(i)
        else:
            strings.append(i)
            idx.append(len(tuples)*offsets + len(strings)-1)

    ori_shape = (len(tuples), ) + reshape_from
    ori_arr = np.array(tuples).reshape(ori_shape)

    new_shape = (offsets*len(tuples), ) + reshape_to
    new_arr = np.transpose(ori_arr, transpose_axes).reshape(new_shape)

    tuples = list(map(tuple, new_arr.reshape(-1, np.prod(reshape_to))))

    merged = []
    tuple_idx = 0
    string_idx = 0
    for i in range(len(tuples) + len(strings)):
        if i in idx:
            merged.append(strings[string_idx])
            string_idx += 1
        else:
            merged.append(tuples[tuple_idx])
            tuple_idx += 1

    return merged


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
        pair = (tuple_list[i], tuple_list[i+1])
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
        if i < len(tuple_list) - 1 and (tuple_list[i], tuple_list[i+1]) == vocab:
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
