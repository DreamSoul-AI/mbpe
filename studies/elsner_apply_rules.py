"""Apply Elsner notebook merge rules without editing the official files.

The replacement order matches processor_python.ipynb: for each saved rule,
scan row then column, look at the downward and rightward neighbour, and
merge when the classes and the anchor offset match. New class ids start
at 256 and increase by one per rule, which is how that notebook numbered
them after seeing raw values 0..255.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "8")

import numpy
import torch

torch.set_num_threads(8)


def initial_state(image: torch.Tensor):
    """image is a long H×W grid of grey levels. Ids follow the notebook."""
    height, width = image.shape
    unique = torch.empty(height, width, dtype=torch.long)
    for x in range(height):
        for y in range(width):
            unique[x, y] = x * height + y
    return unique, image.clone()


def apply_one_rule(unique, classes, class_a, class_b, offset_x, offset_y, new_id):
    """One official replacement pass. Returns how many tokens were merged.

    Candidates are gathered first, then accepted in the notebook's scan
    order (row, column, downward neighbour, then rightward neighbour).
    A token merged earlier in the same pass is skipped, which matches the
    notebook marking that unique id as already replaced.
    """
    grid = unique.numpy()
    values = classes.numpy()
    height, width = grid.shape
    flat = numpy.ascontiguousarray(grid).reshape(-1)
    first = numpy.unique(flat, return_index=True)[1]
    anchor_x = {}
    anchor_y = {}
    for index in first.tolist():
        token_id = int(flat[index])
        anchor_x[token_id] = index // width
        anchor_y[token_id] = index % width

    candidates = []
    if height > 1:
        mask = (values[:-1, :] == class_a) & (values[1:, :] == class_b) & (grid[:-1, :] != grid[1:, :])
        rows, cols = numpy.nonzero(mask)
        for x, y in zip(rows.tolist(), cols.tolist()):
            candidates.append((x, y, 0, x + 1, y))
    if width > 1:
        mask = (values[:, :-1] == class_a) & (values[:, 1:] == class_b) & (grid[:, :-1] != grid[:, 1:])
        rows, cols = numpy.nonzero(mask)
        for x, y in zip(rows.tolist(), cols.tolist()):
            candidates.append((x, y, 1, x, y + 1))
    candidates.sort()

    used = set()
    merged = 0
    for x, y, _order, n_x, n_y in candidates:
        token_id = int(grid[x, y])
        neighbour_id = int(grid[n_x, n_y])
        if token_id in used or neighbour_id in used or token_id == neighbour_id:
            continue
        if int(values[x, y]) != class_a or int(values[n_x, n_y]) != class_b:
            continue
        if anchor_x[token_id] - anchor_x[neighbour_id] != offset_x:
            continue
        if anchor_y[token_id] - anchor_y[neighbour_id] != offset_y:
            continue
        grid[grid == neighbour_id] = token_id
        values[grid == token_id] = new_id
        used.add(token_id)
        used.add(neighbour_id)
        merged += 1
    return merged


def apply_rules(image: torch.Tensor, rules, first_new_id: int = 256):
    unique, classes = initial_state(image)
    for index, rule in enumerate(rules):
        class_a, class_b, offset_x, offset_y = (int(part) for part in rule)
        apply_one_rule(unique, classes, class_a, class_b, offset_x, offset_y, first_new_id + index)
    return unique, classes


def sequence_length(unique: torch.Tensor) -> int:
    return int(torch.unique(unique).numel())


def paint_original(original: torch.Tensor, unique: torch.Tensor) -> torch.Tensor:
    """Place each token's stored original grey values back onto the grid."""
    painted = torch.full_like(original, -1)
    height, width = original.shape
    seen = set()
    for x in range(height):
        for y in range(width):
            token_id = int(unique[x, y])
            if token_id in seen:
                continue
            seen.add(token_id)
            pixels = torch.nonzero(unique == token_id)
            for index in range(pixels.shape[0]):
                px = int(pixels[index, 0])
                py = int(pixels[index, 1])
                if int(painted[px, py]) != -1:
                    raise RuntimeError("token footprints overlap")
                painted[px, py] = original[px, py]
    if int((painted == -1).sum()) != 0:
        raise RuntimeError("token footprints do not cover the image")
    return painted
