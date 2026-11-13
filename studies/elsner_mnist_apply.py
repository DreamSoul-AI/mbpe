"""Apply the batch-16 MNIST merge rules to a train subset and a test subset.

Does not modify the official notebook or the saved rules. The train subset
is the same first-50-per-class selection used when those rules were learned.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "8")

import torch

torch.set_num_threads(8)
try:
    torch.set_num_interop_threads(8)
except RuntimeError:
    pass

from torchvision import datasets, transforms

from elsner_apply_rules import apply_rules, paint_original, sequence_length

REPO = Path(__file__).resolve().parents[1]
MNIST_ROOT = REPO / "third_party" / "MDBPE_TF" / "data"
RULES_PATH = REPO / "results" / "2026-11-10-elsner-mnist-bpe-50" / "rules.dat"
SAVED_ROOT = REPO / "results" / "2026-11-10-elsner-mnist-bpe-50"
OUT = REPO / "results" / "2026-11-10-elsner-mnist-apply"
BASELINE = 28 * 28
PAPER = 54.23


def _grey(image: torch.Tensor) -> torch.Tensor:
    """Same conversion as processor_python.ipynb: ToTensor, then times 255, then int."""
    return (image * 255.0).int()


def _per_class_indices(targets, per_class: int):
    counts = {}
    chosen = []
    for index, label in enumerate(targets.tolist()):
        label = int(label)
        if counts.get(label, 0) < per_class:
            chosen.append(index)
            counts[label] = counts.get(label, 0) + 1
    return chosen


def _summarize(lengths):
    mean = sum(lengths) / len(lengths)
    return {
        "images": len(lengths),
        "baseline_length": BASELINE,
        "mean_length": mean,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "percent_of_original": mean / BASELINE * 100.0,
        "paper_figure6_percent": PAPER,
        "gap_percentage_points": mean / BASELINE * 100.0 - PAPER,
    }


def _check_saved_grids(rules, limit: int = 3):
    """The batch-16 files already store the notebook's final grids."""
    checked = 0
    for path in sorted(SAVED_ROOT.joinpath("transcribed_data_og").rglob("*.dat")):
        relative = path.relative_to(SAVED_ROOT / "transcribed_data_og")
        original = torch.load(path, weights_only=False).long()
        saved_unique, saved_classes = torch.load(
            SAVED_ROOT / "transcribed_data" / relative, weights_only=False
        )
        unique, classes = apply_rules(original, rules)
        if not torch.equal(unique, saved_unique.long()) or not torch.equal(classes, saved_classes.long()):
            raise SystemExit(f"apply does not match the saved notebook grid: {relative}")
        if not torch.equal(paint_original(original, unique), original):
            raise SystemExit(f"saved grid does not rebuild the original image: {relative}")
        checked += 1
        if checked >= limit:
            return checked
    return checked


def _lengths_for(dataset, indices, rules):
    lengths = []
    started = time.time()
    for count, index in enumerate(indices, start=1):
        image, _label = dataset[index]
        grey = _grey(image[0])
        unique, _classes = apply_rules(grey, rules)
        lengths.append(sequence_length(unique))
        if count == 1 or count % 50 == 0:
            spent = time.time() - started
            print(
                f"applied {count}/{len(indices)} mean={sum(lengths)/len(lengths):.3f} "
                f"seconds={spent:.1f}",
                flush=True,
            )
    return lengths, time.time() - started


def _preprocessing_note():
    raw_set = datasets.MNIST(str(MNIST_ROOT), train=True, download=False, transform=None)
    tensor_set = datasets.MNIST(
        str(MNIST_ROOT),
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))]),
    )
    raw = torch.tensor(numpy_image(raw_set[0][0]), dtype=torch.int32)
    converted = _grey(tensor_set[0][0][0])
    mismatch = int((raw != converted).sum())
    return {
        "notebook_path": "ToTensor, Resize(28, 28), times 255, int",
        "raw_uint8_vs_notebook_mismatched_pixels_on_first_train_image": mismatch,
        "raw_min": int(raw.min()),
        "raw_max": int(raw.max()),
        "converted_min": int(converted.min()),
        "converted_max": int(converted.max()),
        "distinct_raw": int(torch.unique(raw).numel()),
        "distinct_converted": int(torch.unique(converted).numel()),
    }


def numpy_image(image):
    import numpy

    return numpy.asarray(image)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rules = torch.load(RULES_PATH, weights_only=False)
    if len(rules) != 256:
        raise SystemExit(f"expected 256 rules, found {len(rules)}")
    print(f"rules={len(rules)} first={rules[0]} last={rules[-1]}", flush=True)
    checked = _check_saved_grids(rules, limit=3)
    print(f"matched {checked} saved notebook grids", flush=True)

    note = _preprocessing_note()
    print("preprocessing", note, flush=True)

    train = datasets.MNIST(
        str(MNIST_ROOT),
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))]),
    )
    test = datasets.MNIST(
        str(MNIST_ROOT),
        train=False,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))]),
    )
    train_index = _per_class_indices(train.targets, 50)
    test_index = _per_class_indices(test.targets, 100)
    train_lengths, train_seconds = _lengths_for(train, train_index, rules)
    test_lengths, test_seconds = _lengths_for(test, test_index, rules)

    full_test = None
    full_seconds = None
    per_image = test_seconds / len(test_index)
    projected_full = per_image * len(test)
    print(f"projected full test seconds={projected_full:.1f}", flush=True)
    if projected_full <= 3600:
        full_lengths, full_seconds = _lengths_for(test, list(range(len(test))), rules)
        full_test = _summarize(full_lengths)

    summary = {
        "rules_path": str(RULES_PATH),
        "rule_count": len(rules),
        "rules_learned_on": "MNIST train, first 50 images per class, official notebook loop",
        "checked_against_saved_notebook_grids": checked,
        "preprocessing": note,
        "length_definition": {
            "counted": "number of remaining unique tokens",
            "stored_entry": "[class, x, y] but x and y are not extra counted tokens",
            "includes_separator": False,
            "includes_padding": False,
            "includes_eos": False,
            "initial_length": BASELINE,
            "compression": "remaining length / 784 * 100",
        },
        "train_subset_self": _summarize(train_lengths),
        "train_subset_seconds": train_seconds,
        "test_100_per_class": _summarize(test_lengths),
        "test_100_per_class_seconds": test_seconds,
        "test_full": full_test,
        "test_full_seconds": full_seconds,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
