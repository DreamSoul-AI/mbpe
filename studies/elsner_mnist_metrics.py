"""Compare MNIST compression definitions without touching official code.

Uses the lengths already measured for the C++ rules and the notebook
rules, then reruns the C++ rules on two altered grey encodings.
"""

from __future__ import annotations

import json
import math
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
os.environ.setdefault("MPLBACKEND", "Agg")

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
OUT = REPO / "results" / "2026-11-10-elsner-mnist-metrics"
PAPER = 54.23
ORIGINAL = 784
RULE_INTS = 4


def _percent(length: float, original: float = ORIGINAL) -> float:
    return length / original * 100.0


def _row(name: str, length: float, note: str) -> dict:
    percent = _percent(length)
    return {
        "name": name,
        "value_percent": percent,
        "distance_to_54_23": abs(percent - PAPER),
        "note": note,
    }


def _formulas(label: str, mean_length: float, images: int) -> list[dict]:
    vocab = TOKEN_MAX + 1 + EXTRA
    bits_per_token = math.log2(vocab)
    original_bits = ORIGINAL * 8
    encoded_bits = mean_length * bits_per_token
    corpus_tokens = mean_length * images
    rows = [
        _row(f"{label} 剩余长度 / 784", mean_length, "论文正文的定义"),
        _row(f"{label} 1 减去剩余比例", ORIGINAL - mean_length, "把压缩理解成删掉的比例"),
        _row(f"{label} 每个 token 按 3 个整数", mean_length * 3, "class、x、y 各算一个"),
        _row(
            f"{label} 比特：长度乘 log2(512) 再除以 784*8",
            encoded_bits / original_bits * ORIGINAL,
            f"实际比特比是 {encoded_bits / original_bits * 100:.4f}%",
        ),
        _row(
            f"{label} 语料总长度加 512 个词表符号，再除以总像素",
            (corpus_tokens + vocab) / images,
            "词表只加一次，分到每张图上",
        ),
        _row(
            f"{label} 每张图再加 256 条规则的 4 个整数",
            mean_length + EXTRA * RULE_INTS,
            "把规则表重复算进每一张图",
        ),
        _row(f"{label} 序列再加开始和结束两个符号", mean_length + 2, "训练时才会加结束符，这里只作对照"),
    ]
    return rows


def _learn_test(train_images: numpy.ndarray, test_images: numpy.ndarray) -> float:
    classes, ids, ax, ay, next_id = _empty_state(train_images)
    shapes = _base_shapes()
    rules = []
    for _step in range(EXTRA):
        key, _count, _ties = most_common(classes, ids, ax, ay)
        shape = _combine(shapes, *key)
        shapes.append(shape)
        apply_rule(classes, ids, ax, ay, next_id, *key, shape, len(shapes) - 1)
        rules.append((key, shape))
    test_classes, test_ids, test_ax, test_ay, test_next = _empty_state(test_images)
    for step, (key, shape) in enumerate(rules):
        apply_rule(
            test_classes,
            test_ids,
            test_ax,
            test_ay,
            test_next,
            key[0],
            key[1],
            key[2],
            key[3],
            shape,
            TOKEN_MAX + 1 + step,
        )
    return float(token_lengths(test_ax, test_ay).mean())


def main():
    started = time.time()
    cpp = json.loads((REPO / "results" / "2026-11-10-elsner-cpp-mnist" / "summary.json").read_text(encoding="utf-8"))
    notebook = json.loads((REPO / "results" / "2026-11-10-elsner-mnist-apply" / "summary.json").read_text(encoding="utf-8"))
    rows = []
    rows.extend(_formulas("C++ 全量测试集", cpp["test_checkpoints"]["256"]["mean_length"], 10000))
    rows.extend(_formulas("笔记本全量测试集", notebook["test_full"]["mean_length"], 10000))

    train_images, train_labels = _load_split(True)
    test_images, _labels = _load_split(False)
    subset = train_images[_per_class(train_labels, 50)]
    altered = {
        "二值化，大于 0 记为 1": numpy.where(subset > 0, 1, 0).astype(numpy.int32),
        "灰度整除 26": numpy.floor_divide(subset, 26).astype(numpy.int32),
    }
    altered_test = {
        "二值化，大于 0 记为 1": numpy.where(test_images > 0, 1, 0).astype(numpy.int32),
        "灰度整除 26": numpy.floor_divide(test_images, 26).astype(numpy.int32),
    }
    relearned = {}
    for name, images in altered.items():
        mean_length = _learn_test(images, altered_test[name])
        relearned[name] = {"mean_length": mean_length, "percent_of_original": _percent(mean_length)}
        rows.append(_row(f"重学后 {name}", mean_length, "仍是剩余长度除以 784，每类 50 张学习，全量测试集"))
        print(name, relearned[name], flush=True)

    rows.sort(key=lambda item: item["distance_to_54_23"])
    summary = {
        "paper_percent": PAPER,
        "closest": rows[0],
        "rows": rows,
        "relearned": relearned,
        "seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
