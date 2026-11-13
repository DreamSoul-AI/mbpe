"""Learn MNIST merges with the official C++ count and replace rules.

The notebook is not modified. Hyperparameters follow the paper's MNIST
row (28 by 28 greyscale, 256 extra tokens), not main.cc, which hardcodes
12 by 12 and 32 new tokens.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("MPLBACKEND", "Agg")

import torch

torch.set_num_threads(8)
try:
    torch.set_num_interop_threads(8)
except RuntimeError:
    pass

import numpy
from torchvision import datasets

REPO = Path(__file__).resolve().parents[1]
MNIST_ROOT = REPO / "third_party" / "MDBPE_TF" / "data"
OUT = REPO / "results" / "2026-11-10-elsner-cpp-mnist"
PAPER = 54.23
EXTRA = 256
TOKEN_MAX = 255


def _keep_source_anchor(offset_x: int, offset_y: int) -> bool:
    return not (offset_y < 0 or (offset_y == 0 and offset_x < 0))


def _empty_state(images: numpy.ndarray):
    count, height, width = images.shape
    classes = numpy.ascontiguousarray(images.astype(numpy.int32))
    ids = numpy.empty((count, height, width), dtype=numpy.int32)
    ax = numpy.empty((count, height, width), dtype=numpy.int32)
    ay = numpy.empty((count, height, width), dtype=numpy.int32)
    yy, xx = numpy.meshgrid(numpy.arange(height), numpy.arange(width), indexing="ij")
    ids[:] = yy * width + xx
    ax[:] = xx
    ay[:] = yy
    next_id = numpy.full(count, height * width, dtype=numpy.int32)
    return classes, ids, ax, ay, next_id


def _base_shapes():
    shapes = []
    for class_id in range(TOKEN_MAX + 1):
        shapes.append([(0, 0, class_id)])
    return shapes


def _combine(shapes, source_class, target_class, offset_x, offset_y):
    offset = (offset_x, offset_y)
    left = shapes[source_class]
    right = shapes[target_class]
    if _keep_source_anchor(offset_x, offset_y):
        merged = list(left)
        merged.extend((dx + offset[0], dy + offset[1], value) for dx, dy, value in right)
    else:
        merged = list(right)
        merged.extend((dx - offset[0], dy - offset[1], value) for dx, dy, value in left)
    return merged


def _boundary_pairs(classes, ids, ax, ay, vertical: bool):
    if vertical:
        source_id = ids[:, :-1, :]
        target_id = ids[:, 1:, :]
        valid = source_id != target_id
        image, y_coord, x_coord = numpy.nonzero(valid)
        source_class = classes[:, :-1, :][valid]
        target_class = classes[:, 1:, :][valid]
        source_ax = ax[:, :-1, :][valid]
        source_ay = ay[:, :-1, :][valid]
        target_ax = ax[:, 1:, :][valid]
        target_ay = ay[:, 1:, :][valid]
        direction = numpy.zeros(image.shape[0], dtype=numpy.int32)
    else:
        source_id = ids[:, :, :-1]
        target_id = ids[:, :, 1:]
        valid = source_id != target_id
        image, y_coord, x_coord = numpy.nonzero(valid)
        source_class = classes[:, :, :-1][valid]
        target_class = classes[:, :, 1:][valid]
        source_ax = ax[:, :, :-1][valid]
        source_ay = ay[:, :, :-1][valid]
        target_ax = ax[:, :, 1:][valid]
        target_ay = ay[:, :, 1:][valid]
        direction = numpy.ones(image.shape[0], dtype=numpy.int32)
    return (
        image,
        y_coord,
        x_coord,
        direction,
        source_class,
        target_class,
        source_ax,
        source_ay,
        target_ax,
        target_ay,
    )


def most_common(classes, ids, ax, ay, skip_zero=False):
    """One C++ counting pass. Ties keep the lexicographically smallest key.

    skip_zero drops any pair that still has the original grey value 0.
    That is not in the paper. New token ids start at 256, so they are kept.
    """
    parts = [_boundary_pairs(classes, ids, ax, ay, True), _boundary_pairs(classes, ids, ax, ay, False)]
    image = numpy.concatenate([part[0] for part in parts])
    order = numpy.lexsort(
        (
            numpy.concatenate([part[3] for part in parts]),
            numpy.concatenate([part[2] for part in parts]),
            numpy.concatenate([part[1] for part in parts]),
            image,
        )
    )
    counts = Counter()
    seen = set()
    source_class = numpy.concatenate([part[4] for part in parts])
    target_class = numpy.concatenate([part[5] for part in parts])
    source_ax = numpy.concatenate([part[6] for part in parts])
    source_ay = numpy.concatenate([part[7] for part in parts])
    target_ax = numpy.concatenate([part[8] for part in parts])
    target_ay = numpy.concatenate([part[9] for part in parts])
    for index in order.tolist():
        image_index = int(image[index])
        left = (int(source_ax[index]), int(source_ay[index]))
        right = (int(target_ax[index]), int(target_ay[index]))
        pair = (image_index, left, right)
        reverse = (image_index, right, left)
        if pair in seen or reverse in seen:
            continue
        seen.add(pair)
        source = int(source_class[index])
        target = int(target_class[index])
        if skip_zero and (source == 0 or target == 0):
            continue
        offset_x = right[0] - left[0]
        offset_y = right[1] - left[1]
        key = (source, target, offset_x, offset_y)
        counts[key] += 1
    if not counts:
        raise RuntimeError("no adjacent pair to merge")
    best_count = max(counts.values())
    winners = [key for key, count in counts.items() if count == best_count]
    winners.sort()
    return winners[0], best_count, len(winners)


def apply_rule(classes, ids, ax, ay, next_id, source_class, target_class, offset_x, offset_y, shape, new_class):
    count, height, width = classes.shape
    yy, xx = numpy.meshgrid(numpy.arange(height), numpy.arange(width), indexing="ij")
    is_anchor = (ax == xx) & (ay == yy)
    image, y_coord, x_coord = numpy.nonzero(is_anchor & (classes == source_class))
    if image.size == 0:
        return 0
    target_y = y_coord + offset_y
    target_x = x_coord + offset_x
    inside = (target_y >= 0) & (target_y < height) & (target_x >= 0) & (target_x < width)
    image = image[inside]
    y_coord = y_coord[inside]
    x_coord = x_coord[inside]
    target_y = target_y[inside]
    target_x = target_x[inside]
    if image.size == 0:
        return 0
    order = numpy.lexsort((x_coord, y_coord, image))
    keep_source = _keep_source_anchor(offset_x, offset_y)
    used = set()
    merged = 0
    for index in order.tolist():
        image_index = int(image[index])
        source_y = int(y_coord[index])
        source_x = int(x_coord[index])
        other_y = int(target_y[index])
        other_x = int(target_x[index])
        if int(classes[image_index, source_y, source_x]) != source_class:
            continue
        if int(classes[image_index, other_y, other_x]) != target_class:
            continue
        if int(ax[image_index, other_y, other_x]) != other_x or int(ay[image_index, other_y, other_x]) != other_y:
            continue
        source_id = int(ids[image_index, source_y, source_x])
        target_id = int(ids[image_index, other_y, other_x])
        if source_id == target_id:
            continue
        if (image_index, source_id) in used or (image_index, target_id) in used:
            continue
        if keep_source:
            anchor_x, anchor_y = source_x, source_y
        else:
            anchor_x, anchor_y = other_x, other_y
        new_id = int(next_id[image_index])
        next_id[image_index] = new_id + 1
        for dx, dy, _value in shape:
            pixel_x = anchor_x + dx
            pixel_y = anchor_y + dy
            if pixel_x < 0 or pixel_y < 0 or pixel_x >= width or pixel_y >= height:
                continue
            classes[image_index, pixel_y, pixel_x] = new_class
            ids[image_index, pixel_y, pixel_x] = new_id
            ax[image_index, pixel_y, pixel_x] = anchor_x
            ay[image_index, pixel_y, pixel_x] = anchor_y
        used.add((image_index, source_id))
        used.add((image_index, target_id))
        merged += 1
    return merged


def token_lengths(ax, ay):
    count, height, width = ax.shape
    yy, xx = numpy.meshgrid(numpy.arange(height), numpy.arange(width), indexing="ij")
    anchors = (ax == xx) & (ay == yy)
    return anchors.reshape(count, -1).sum(axis=1)


def paint_original(shapes, classes, ax, ay, originals):
    painted = numpy.full_like(originals, -1)
    count, height, width = classes.shape
    for image_index in range(count):
        seen = set()
        for y_coord in range(height):
            for x_coord in range(width):
                token_id = int(classes[image_index, y_coord, x_coord])
                anchor_x = int(ax[image_index, y_coord, x_coord])
                anchor_y = int(ay[image_index, y_coord, x_coord])
                if (anchor_x, anchor_y) in seen:
                    continue
                seen.add((anchor_x, anchor_y))
                for dx, dy, value in shapes[token_id]:
                    pixel_x = anchor_x + dx
                    pixel_y = anchor_y + dy
                    if pixel_x < 0 or pixel_y < 0 or pixel_x >= width or pixel_y >= height:
                        continue
                    if painted[image_index, pixel_y, pixel_x] not in (-1, value):
                        raise RuntimeError("overlap while rebuilding")
                    painted[image_index, pixel_y, pixel_x] = value
    if int((painted < 0).sum()) != 0:
        raise RuntimeError("rebuild missed pixels")
    if not numpy.array_equal(painted, originals):
        raise RuntimeError("rebuild does not match the original grey values")


def _self_check():
    image = numpy.zeros((1, 4, 4), dtype=numpy.int32)
    classes, ids, ax, ay, next_id = _empty_state(image)
    shapes = _base_shapes()
    key, count, ties = most_common(classes, ids, ax, ay)
    if key != (0, 0, 0, 1) or count != 12:
        raise RuntimeError(f"self-check count failed: {key} {count} ties={ties}")
    shape = _combine(shapes, *key)
    shapes.append(shape)
    apply_rule(classes, ids, ax, ay, next_id, *key, shape, len(shapes) - 1)
    lengths = token_lengths(ax, ay)
    if int(lengths[0]) != 8:
        raise RuntimeError(f"self-check length failed: {int(lengths[0])}")
    paint_original(shapes, classes, ax, ay, image)


def _load_split(train: bool):
    dataset = datasets.MNIST(str(MNIST_ROOT), train=train, download=False)
    images = numpy.empty((len(dataset), 28, 28), dtype=numpy.int32)
    labels = numpy.empty(len(dataset), dtype=numpy.int32)
    for index in range(len(dataset)):
        grey, label = dataset[index]
        images[index] = numpy.asarray(grey, dtype=numpy.int32)
        labels[index] = int(label)
    return images, labels


def _per_class(labels, per_class: int):
    chosen = []
    counts = {}
    for index, label in enumerate(labels.tolist()):
        if counts.get(label, 0) < per_class:
            chosen.append(index)
            counts[label] = counts.get(label, 0) + 1
    return numpy.asarray(chosen, dtype=numpy.int32)


def _summarize(lengths):
    mean = float(lengths.mean())
    return {
        "images": int(lengths.size),
        "mean_length": mean,
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "percent_of_original": mean / 784.0 * 100.0,
        "paper_percent": PAPER,
    }


def main():
    started = time.time()
    _self_check()
    print("self-check ok", flush=True)
    train_images, train_labels = _load_split(True)
    test_images, _test_labels = _load_split(False)
    subset = _per_class(train_labels, 50)
    learn_images = train_images[subset]
    print(f"learn_images={learn_images.shape[0]} test_images={test_images.shape[0]}", flush=True)

    classes, ids, ax, ay, next_id = _empty_state(learn_images)
    shapes = _base_shapes()
    rules = []
    for step in range(EXTRA):
        key, count, ties = most_common(classes, ids, ax, ay)
        shape = _combine(shapes, *key)
        new_class = len(shapes)
        shapes.append(shape)
        merged = apply_rule(classes, ids, ax, ay, next_id, *key, shape, new_class)
        rules.append({"source": key[0], "target": key[1], "offset_x": key[2], "offset_y": key[3], "count": count, "ties": ties, "merged": merged})
        if step == 0 or step + 1 in (8, 32, 64, 128, 256):
            lengths = token_lengths(ax, ay)
            print(
                f"rule {step + 1} key={key} count={count} ties={ties} merged={merged} "
                f"train_mean={float(lengths.mean()):.3f} seconds={time.time() - started:.1f}",
                flush=True,
            )
    paint_original(shapes, classes[:2], ax[:2], ay[:2], learn_images[:2])
    train_lengths = token_lengths(ax, ay)

    test_classes, test_ids, test_ax, test_ay, test_next = _empty_state(test_images)
    checkpoints = {}
    for step, rule in enumerate(rules):
        apply_rule(
            test_classes,
            test_ids,
            test_ax,
            test_ay,
            test_next,
            rule["source"],
            rule["target"],
            rule["offset_x"],
            rule["offset_y"],
            shapes[TOKEN_MAX + 1 + step],
            TOKEN_MAX + 1 + step,
        )
        if step < 8 or step + 1 in (32, 64, 128, 256):
            lengths = token_lengths(test_ax, test_ay)
            checkpoints[str(step + 1)] = _summarize(lengths)
            print(f"test after {step + 1} rules {checkpoints[str(step + 1)]}", flush=True)
    paint_original(shapes, test_classes[:2], test_ax[:2], test_ay[:2], test_images[:2])

    summary = {
        "algorithm": "third_party/MDBPE_TF/MDBPE/src/tokenizer.cc count and replace",
        "tie_break": "highest count, then lexicographic (source, target, offset_x, offset_y)",
        "learn_images": "MNIST train, first 50 per class",
        "extra_tokens": EXTRA,
        "initial_length": 784,
        "first_rule": rules[0],
        "train_subset": _summarize(train_lengths),
        "test_checkpoints": checkpoints,
        "seconds": time.time() - started,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
