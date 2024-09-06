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

    data = np.squeeze(np.array(data))
    assert len(data.shape) <= 3, "Data must be equal to or less than 3D"

    if len(data.shape) == 2:
        rows, cols = data.shape
    else:
        raise ValueError("Havn't implemented for 3D data yet")

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
            for r_idx in row_indices_group:
                for c_idx in col_indices_group:
                    group.append(data[r_idx, c_idx])
            tuples.append(tuple(group))

    return tuples


def reshape_tuples(data, reshape_from, reshape_to):
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

    new = []
    for i in data:
        if isinstance(i, tuple):
            row_group_size = reshape_from[0] // reshape_to[0]
            col_group_size = reshape_from[1] // reshape_to[1]

            row_indices = []
            col_indices = []
            for j in range(0, row_group_size):
                row_indices.append(
                    [k for k in range(j, reshape_from[0], row_group_size)])

            for j in range(0, col_group_size):
                col_indices.append(
                    [k for k in range(j, reshape_from[1], col_group_size)])

            for row_indices_group in row_indices:
                for col_indices_group in col_indices:
                    group = []
                    for r_idx in row_indices_group:
                        for c_idx in col_indices_group:
                            group.append(i[r_idx*reshape_from[1]+c_idx])
                    new.append(tuple(group))
        else:
            new.append(i)
    return new


def reshape_tuple(data):
    """
    Reshape the given data into different dim of tuples.

    Args:
        - dim (2, 1) tuple list → dim (1, 2) tuple list
        - dim (1, 2) tuple list → dim (1, 1) tuple list

    Returns:
        tuples: A list of tuples generated through reshaping the data.

    """

    reshaped_data = []
    if any(isinstance(item, tuple) and len(item) == 4 for item in data):  # dim (2, 2) → dim (1, 2)
        for item in data:
            if isinstance(item, tuple):
                reshaped_data.extend([(item[i], item[i + 1]) for i in range(0, len(item), 2)])
            else:
                reshaped_data.append(item)

    else:  # dim (1, 2) → dim (1, 1)
        for item in data:
            if isinstance(item, tuple):
                for sub_item in item:
                    reshaped_data.append((sub_item,))
            else:
                reshaped_data.append(item)

    return reshaped_data


def freq_tuple(tuple_list):
    freq = defaultdict(int)
    for t in tuple_list:
        freq[t] += 1
    return freq


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


def merge(tuple_list, root_vocab, max_pair):

    if max_pair not in root_vocab:
        root_vocab.append(max_pair)
    

    new_tuple_list = []
    skip_next = False  
    for i in range(len(tuple_list)):
        if skip_next:
            skip_next = False
            continue

        if i < len(tuple_list) - 1 and tuple_list[i] == max_pair[0] and tuple_list[i + 1] == max_pair[1]:
            new_tuple_list.append(max_pair)
            skip_next = True  
        else:
            new_tuple_list.append(tuple_list[i])

    return new_tuple_list, root_vocab


def dfs(pair, vocab):
    """
    Perform a depth-first search (DFS) to decode a pair recursively using the provided vocabulary.

    Args:
        pair (tuple): A tuple representing a pair of tuples or string codes to be decoded.
        vocab (dict): A dictionary mapping string codes to tuples representing pairs.

    Returns:
        list: A list of decoded elements resulting from the DFS decoding process.

    Notes:
        This function decodes a pair recursively by traversing the vocabulary using depth-first search (DFS).
        It handles both elements of the pair, decoding them until they are no longer represented by string codes.
    """

    pair = list(pair)
    while isinstance(pair[0], str):
        pair[0] = dfs(vocab[pair[0]], vocab)

    while isinstance(pair[1], str):
        pair[1] = dfs(vocab[pair[1]], vocab)

    if isinstance(pair[0], tuple):
        pair[0] = [pair[0]]
    if isinstance(pair[1], tuple):
        pair[1] = [pair[1]]
    return pair[0] + pair[1]


def compression_rate(data, encoded_data):
    """
    Computes the compression rate of the encoded data.

    Args:
        data (torch.Tensor): The original data.
        encoded_data (numpy.ndarray): The encoded data.

    Returns:
        rate (float): The compression rate.
    """

    original_size = data.flatten().shape[0] * 8
    encoded_size = 0
    for i in encoded_data:
        if isinstance(i, str):
            encoded_size += 8
        else:
            encoded_size += 8 * len(i)
    return encoded_size / original_size
