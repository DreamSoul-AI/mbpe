import numpy as np
import numbers
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


def dfs(input, search_fn):
    if isinstance(input, numbers.Integral):
        return [input]
    elif isinstance(input, str):
        return dfs(search_fn(input), search_fn)
    elif isinstance(input, tuple):
        result = []
        for item in input:
            result.extend(dfs(item, search_fn))
        return result
    else:
        raise TypeError(f"Unexpected type {type(input)} in dfs")


def compress_indices(indices):  # TODO: need better compression
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
