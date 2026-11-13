"""Run the official Python tokenizer on MNIST without editing its files.

The official notebook hard-codes 12x12 and 32 extra tokens. This wrapper
executes those cells from a memory copy, after substituting the paper
settings and an optional per-class subset. The notebook on disk is not
written. C++ was not built: this machine has no CMake or C++ compiler,
and the official CMake target also links GLFW and Glow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "8")

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "third_party" / "MDBPE_TF" / "MDBPE" / "processor_python.ipynb"
MNIST_ROOT = REPO / "third_party" / "MDBPE_TF" / "data"


def _load_cells():
    import json as _json

    notebook = _json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cells.append("".join(cell.get("source", [])))
    return cells


def _prepare(source: str, extra_tokens: int, per_class: int) -> str:
    source = source.replace("WIDTH, HEIGHT = 12, 12", "WIDTH, HEIGHT = 28, 28", 1)
    source = source.replace(
        "MAXIMUM_NUMBER_OF_TOKENS = 256+32 #256 = only use the base tokens, 256+32 --> use 32 extra tokens to compact stuff into",
        f"MAXIMUM_NUMBER_OF_TOKENS = 256+{extra_tokens} #256 = only use the base tokens, 256+32 --> use 32 extra tokens to compact stuff into",
        1,
    )
    source = source.replace(
        'LIMITED_EXAMPLES = 100 #--> only use a subset of the data (for debugging); set to "None" to use all data',
        'LIMITED_EXAMPLES = None #--> only use a subset of the data (for debugging); set to "None" to use all data',
        1,
    )
    source = source.replace("root='./data'", f"root=r'{MNIST_ROOT}'", 1)
    if "train_loader = DataLoader(mnist_trainset, batch_size=8, shuffle=False)" in source:
        source += f"""

_counts = {{}}
_idx = []
for _i, _y in enumerate(mnist_trainset.targets.tolist()):
    _y = int(_y)
    if _counts.get(_y, 0) < {int(per_class)}:
        _idx.append(_i)
        _counts[_y] = _counts.get(_y, 0) + 1
mnist_trainset = torch.utils.data.Subset(mnist_trainset, _idx)
train_loader = DataLoader(mnist_trainset, batch_size=8, shuffle=False, num_workers=0)
test_loader = DataLoader(mnist_testnset, batch_size=8, shuffle=False, num_workers=0)
"""
    return source


def _summarize(work: Path, width: int, height: int) -> dict:
    import torch

    files = sorted((work / "tokensequences").rglob("*.dat"))
    lengths = [len(torch.load(path, weights_only=False)) for path in files]
    baseline = width * height
    mean = sum(lengths) / len(lengths) if lengths else None
    return {
        "images": len(lengths),
        "baseline_tokens": baseline,
        "mean_length": mean,
        "min_length": min(lengths) if lengths else None,
        "max_length": max(lengths) if lengths else None,
        "compression_percent_of_original": (mean / baseline * 100.0) if mean else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--extra-tokens", type=int, default=256)
    parser.add_argument("--per-class", type=int, default=50)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(8)
    try:
        torch.set_num_interop_threads(8)
    except RuntimeError:
        pass

    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    print(f"work={work}", flush=True)
    print(f"notebook={NOTEBOOK}", flush=True)
    print(f"extra_tokens={args.extra_tokens} per_class={args.per_class}", flush=True)
    print(f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    namespace: dict = {"__name__": "__main__"}
    stop_after = "DONE EXPORTING "
    for index, source in enumerate(_load_cells()):
        if source.strip().startswith("DUMMY_CASE"):
            break
        prepared = _prepare(source, args.extra_tokens, args.per_class)
        print(f"--- cell {index} ({len(prepared)} chars) ---", flush=True)
        exec(compile(prepared, f"{NOTEBOOK}::cell{index}", "exec"), namespace, namespace)
        if "sequence_lengths" in namespace and stop_after in prepared:
            break

    summary = _summarize(work, 28, 28)
    summary["extra_tokens"] = args.extra_tokens
    summary["per_class"] = args.per_class
    summary["paper_figure6_mnist_compression_percent"] = 54.23
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
