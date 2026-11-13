"""Grayscale and per-channel runs on CIFAR-10, SVHN and CelebA.

The official notebook only stores channel 0. It is not changed. This script
turns each picture into one 0-255 grid before that notebook sees it.
Setting A is PIL's L conversion. Setting B runs red, green and blue as
three separate grids, because the notebook has no multi-channel grid.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"

import numpy
from PIL import Image

import elsner_svhn_celeba as runner

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "results" / "2026-11-10-elsner-color-modes"
EACH = 10
EXTRA = 256
SIZE = 32


def _per_class(dataset, per_class: int, classes: int = 10):
    counts = {}
    chosen = []
    for index in range(len(dataset)):
        image, label = dataset[index]
        label = int(label)
        if counts.get(label, 0) >= per_class:
            continue
        counts[label] = counts.get(label, 0) + 1
        chosen.append(image.convert("RGB"))
        if len(counts) >= classes and min(counts.values()) >= per_class:
            break
    return chosen


def _gray(image: Image.Image) -> numpy.ndarray:
    return numpy.asarray(image.convert("L"), dtype=numpy.uint8)


def _channel(image: Image.Image, index: int) -> numpy.ndarray:
    return numpy.asarray(image.convert("RGB"), dtype=numpy.uint8)[:, :, index]


def _cifar(train: bool):
    from torchvision import datasets

    return datasets.CIFAR10(str(DATA), train=train, download=False)


def _svhn(split: str):
    from torchvision import datasets

    return datasets.SVHN(str(DATA), split=split, download=False)


def _celeba(start: int, stop: int):
    folder = DATA / "celeba" / "partial_jpg"
    files = sorted(folder.glob("*.jpg"))[start:stop]
    return [Image.open(path).convert("RGB") for path in files]


def _grids(images, mode: str):
    if mode == "gray":
        return [_gray(image) for image in images]
    channel = {"red": 0, "green": 1, "blue": 2}[mode]
    return [_channel(image, channel) for image in images]


def _done(path: Path) -> bool:
    if not path.exists():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    if path.parent.name == "train":
        return int(summary.get("rules", 0)) == EXTRA and int(summary.get("images", 0)) == EACH
    return int(summary.get("images", 0)) == EACH and "compression_percent_of_original" in summary


def _run(name: str, train_images, test_images, mode: str) -> dict:
    root = OUT / name / mode
    train_dir = root / "train"
    test_dir = root / "test"
    train_dir.parent.mkdir(parents=True, exist_ok=True)
    if not _done(train_dir / "summary.json"):
        runner.PREPARED_GRIDS = _grids(train_images, mode)
        runner._learn(train_dir, "grids", 1, EACH, SIZE, SIZE, EXTRA)
    train_summary = json.loads((train_dir / "summary.json").read_text(encoding="utf-8"))
    if not _done(test_dir / "summary.json"):
        runner.PREPARED_GRIDS = _grids(test_images, mode)
        runner._apply_test(test_dir, train_dir, "grids", 1, EACH, SIZE, SIZE, EXTRA)
    test_summary = json.loads((test_dir / "summary.json").read_text(encoding="utf-8"))
    report = {
        "dataset": name,
        "mode": mode,
        "train_images": EACH,
        "test_images": EACH,
        "train_seconds": train_summary["seconds"],
        "test_seconds": test_summary["seconds"],
        "mean_length": test_summary["mean_length"],
        "compression_percent_of_original": test_summary["compression_percent_of_original"],
        "lossless_bad": test_summary["lossless_bad"],
        "lossless_checked": test_summary["lossless_checked"],
        "rules_missed_on_this_split": test_summary.get("rules_missed_on_this_split"),
        "baseline_tokens": test_summary["baseline_tokens"],
    }
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("REPORT", json.dumps(report), flush=True)
    return report


def main() -> int:
    print(f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)
    cifar_train = _per_class(_cifar(True), 1)
    cifar_test = _per_class(_cifar(False), 1)
    svhn_train = _per_class(_svhn("train"), 1)
    svhn_test = _per_class(_svhn("test"), 1)
    celeba_train = _celeba(0, EACH)
    celeba_test = _celeba(EACH, EACH * 2)
    print(
        f"counts cifar {len(cifar_train)}/{len(cifar_test)} "
        f"svhn {len(svhn_train)}/{len(svhn_test)} "
        f"celeba {len(celeba_train)}/{len(celeba_test)}",
        flush=True,
    )
    jobs = [
        ("cifar10", cifar_train, cifar_test),
        ("svhn", svhn_train, svhn_test),
        ("celeba", celeba_train, celeba_test),
    ]
    modes = ("gray", "red", "green", "blue")
    reports = []
    for name, train_images, test_images in jobs:
        for mode in modes:
            print(f"=== {name} {mode} ===", flush=True)
            reports.append(_run(name, train_images, test_images, mode))
    out = OUT / "summary.json"
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print("WROTE", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
