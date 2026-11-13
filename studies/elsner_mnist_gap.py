"""MNIST gap checks: skip grey 0, and the length curve from 1 to 256 tokens."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "8")
os.environ["MPLBACKEND"] = "Agg"

import torch

torch.set_num_threads(8)
try:
    torch.set_num_interop_threads(8)
except RuntimeError:
    pass

import numpy

from elsner_cpp_mnist import (
    EXTRA,
    TOKEN_MAX,
    _base_shapes,
    _combine,
    _empty_state,
    _load_split,
    _per_class,
    apply_rule,
    most_common,
    token_lengths,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "2026-11-10-elsner-mnist-gap"
PAPER = 54.23
ORIGINAL = 784


def _curve(train_images, test_images, skip_zero: bool):
    classes, ids, ax, ay, next_id = _empty_state(train_images)
    shapes = _base_shapes()
    rules = []
    for _step in range(EXTRA):
        try:
            key, count, ties = most_common(classes, ids, ax, ay, skip_zero=skip_zero)
        except RuntimeError:
            break
        shape = _combine(shapes, *key)
        shapes.append(shape)
        apply_rule(classes, ids, ax, ay, next_id, *key, shape, len(shapes) - 1)
        rules.append({"key": key, "count": count, "ties": ties, "shape": shape})
    test_classes, test_ids, test_ax, test_ay, test_next = _empty_state(test_images)
    percents = []
    lengths = None
    for step, rule in enumerate(rules):
        apply_rule(
            test_classes,
            test_ids,
            test_ax,
            test_ay,
            test_next,
            rule["key"][0],
            rule["key"][1],
            rule["key"][2],
            rule["key"][3],
            rule["shape"],
            TOKEN_MAX + 1 + step,
        )
        lengths = token_lengths(test_ax, test_ay)
        percents.append(float(lengths.mean()) / ORIGINAL * 100.0)
    ink_percent = None
    if lengths is not None and not skip_zero:
        ink = _ink_lengths(test_classes, test_ax, test_ay, shapes)
        ink_percent = float(ink.mean()) / ORIGINAL * 100.0
    nearest = None
    if percents:
        distances = [abs(value - PAPER) for value in percents]
        index = int(numpy.argmin(distances))
        nearest = {
            "extra_tokens": index + 1,
            "percent": percents[index],
            "distance": distances[index],
        }
    return {
        "rules_learned": len(rules),
        "first_rule": rules[0]["key"] if rules else None,
        "percents": percents,
        "final_percent": percents[-1] if percents else None,
        "nearest_to_54_23": nearest,
        "ink_only_percent": ink_percent,
    }


def _ink_lengths(classes, ax, ay, shapes):
    count, height, width = classes.shape
    totals = numpy.zeros(count, dtype=numpy.int32)
    for image_index in range(count):
        seen = set()
        for y_coord in range(height):
            for x_coord in range(width):
                anchor = (int(ax[image_index, y_coord, x_coord]), int(ay[image_index, y_coord, x_coord]))
                if anchor in seen:
                    continue
                seen.add(anchor)
                class_id = int(classes[image_index, y_coord, x_coord])
                if any(value != 0 for _dx, _dy, value in shapes[class_id]):
                    totals[image_index] += 1
    return totals


def _plot(standard, skipped):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(range(1, len(standard) + 1), standard, label="paper rules")
    axis.plot(range(1, len(skipped) + 1), skipped, label="skip grey 0")
    axis.axhline(PAPER, color="black", linestyle="--", label="paper 54.23")
    axis.set_xlabel("extra tokens")
    axis.set_ylabel("test length / 784, percent")
    axis.set_xlim(1, 256)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUT / "curve.png", dpi=120)
    plt.close(figure)


def main():
    started = time.time()
    train_images, train_labels = _load_split(True)
    test_images, _labels = _load_split(False)
    subset = train_images[_per_class(train_labels, 50)]
    zero_fraction = float((test_images == 0).mean()) * 100.0
    print(f"test pixels that are 0: {zero_fraction:.2f}%", flush=True)
    standard = _curve(subset, test_images, False)
    print("standard", standard["final_percent"], standard["nearest_to_54_23"], flush=True)
    skipped = _curve(subset, test_images, True)
    print("skip0", skipped["final_percent"], skipped["nearest_to_54_23"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    _plot(standard["percents"], skipped["percents"])
    summary = {
        "paper_percent": PAPER,
        "test_zero_pixel_percent": zero_fraction,
        "standard": {key: value for key, value in standard.items() if key != "percents"},
        "standard_percents": standard["percents"],
        "skip_zero": {key: value for key, value in skipped.items() if key != "percents"},
        "skip_zero_percents": skipped["percents"],
        "seconds": time.time() - started,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("standard", "skip_zero", "test_zero_pixel_percent", "seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
