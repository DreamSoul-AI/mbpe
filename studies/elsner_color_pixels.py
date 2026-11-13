"""Our own color-pixel runs through the official Python tokenizer.

The notebook file is not modified. An in-memory copy only changes width,
height, and how many extra symbols. This script turns each picture into a
0-255 grid before that copy sees it.

Official transcription keeps channel 0 only, so there is no multi-channel
grid. Setting gray is one luminance grid. Settings r, g, and b each feed
one color channel as its own gray grid, in three separate runs.

No quantization weights. Pixels are not VQ-VAE codes.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"

ctypes.windll.kernel32.SetPriorityClass(
    ctypes.windll.kernel32.GetCurrentProcess(),
    0x00004000,
)

import torch

torch.set_num_threads(8)
try:
    torch.set_num_interop_threads(8)
except RuntimeError:
    pass

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elsner_svhn_celeba import (  # noqa: E402
    _cells,
    _lossless,
    _namespace,
    _prepare_setup,
    _replace_block,
)

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "third_party" / "MDBPE_TF" / "MDBPE" / "processor_python.ipynb"
DATA = REPO / "data"
RESULTS = REPO / "results" / "2026-11-10-elsner-color-pixels"
CHANNELS = {"gray": None, "r": 0, "g": 1, "b": 2}


def _subset_source(kind: str, split: str, per_class: int, offset: int, length: int, channel: int | None) -> str:
    root = str(DATA)
    channel_literal = "None" if channel is None else str(int(channel))
    if kind == "cifar":
        train_flag = "True" if split == "train" else "False"
        return f"""
_base = datasets.CIFAR10(root=r'{root}', train={train_flag}, download=False)
_counts = {{}}
_idx = []
for _i, _y in enumerate(list(_base.targets)):
    _y = int(_y)
    if _counts.get(_y, 0) < {int(per_class)}:
        _idx.append(_i)
        _counts[_y] = _counts.get(_y, 0) + 1
class _OneChannel(torch.utils.data.Dataset):
    def __init__(self, base, indices, transform, channel):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.channel = channel
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        image, _label = self.base[self.indices[index]]
        image = image.convert("RGB")
        if self.channel is None:
            image = image.convert("L")
        else:
            image = image.getchannel(int(self.channel))
        if self.transform is not None:
            image = self.transform(image)
        return image, 0
mnist_trainset = _OneChannel(_base, _idx, transform, {channel_literal})
train_loader = DataLoader(mnist_trainset, batch_size=8, shuffle=False, num_workers=0)
test_loader = train_loader
CHOSEN_INDICES = list(_idx)
"""
    if kind == "svhn":
        return f"""
_base = datasets.SVHN(root=r'{root}', split='{split}', download=False)
_labels = _base.labels.reshape(-1)
_counts = {{}}
_idx = []
for _i, _y in enumerate(_labels.tolist()):
    _y = int(_y)
    if _counts.get(_y, 0) < {int(per_class)}:
        _idx.append(_i)
        _counts[_y] = _counts.get(_y, 0) + 1
class _OneChannel(torch.utils.data.Dataset):
    def __init__(self, base, indices, transform, channel):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.channel = channel
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        image, _label = self.base[self.indices[index]]
        image = image.convert("RGB")
        if self.channel is None:
            image = image.convert("L")
        else:
            image = image.getchannel(int(self.channel))
        if self.transform is not None:
            image = self.transform(image)
        return image, 0
mnist_trainset = _OneChannel(_base, _idx, transform, {channel_literal})
train_loader = DataLoader(mnist_trainset, batch_size=8, shuffle=False, num_workers=0)
test_loader = train_loader
CHOSEN_INDICES = list(_idx)
"""
    folder = DATA / "celeba" / "partial_jpg"
    start = int(offset)
    stop = start + int(length)
    return f"""
import PIL
from pathlib import Path
_files = sorted(Path(r'{folder}').glob('*.jpg'))[{start}:{stop}]
class _OneChannel(torch.utils.data.Dataset):
    def __init__(self, files, transform, channel):
        self.files = list(files)
        self.transform = transform
        self.channel = channel
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        image = PIL.Image.open(self.files[index]).convert("RGB")
        if self.channel is None:
            image = image.convert("L")
        else:
            image = image.getchannel(int(self.channel))
        if self.transform is not None:
            image = self.transform(image)
        return image, 0
mnist_trainset = _OneChannel(_files, transform, {channel_literal})
train_loader = DataLoader(mnist_trainset, batch_size=8, shuffle=False, num_workers=0)
test_loader = train_loader
CHOSEN_INDICES = [path.name for path in _files]
"""


def _transcribe(namespace, cells, kind, split, per_class, offset, length, width, height, extra, channel):
    exec(compile(_prepare_setup(cells[1], width, height, extra), f"{NOTEBOOK}::cell1", "exec"), namespace, namespace)
    dataset_cell = cells[2]
    dataset_cell = dataset_cell.replace(
        "mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)\n"
        "mnist_testnset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)\n"
        "train_loader = DataLoader(mnist_trainset, batch_size=8, shuffle=False)\n"
        "test_loader = DataLoader(mnist_testnset, batch_size=8, shuffle=False)\n",
        _subset_source(kind, split, per_class, offset, length, channel),
        1,
    )
    show_at = dataset_cell.find("def show(")
    if show_at != -1:
        dataset_cell = dataset_cell[:show_at] + "pass\n"
    exec(compile(dataset_cell, f"{NOTEBOOK}::cell2", "exec"), namespace, namespace)
    for index in (3, 4, 5, 6, 7, 8, 9):
        exec(compile(cells[index], f"{NOTEBOOK}::cell{index}", "exec"), namespace, namespace)
    namespace["visualise_unique"] = lambda *_args, **_kwargs: None
    namespace["NEW_TOKEN_ID"] = 256
    namespace["START_OF_NEW_TOKENS"] = 256
    namespace["MAXIMUM_NUMBER_OF_TOKENS"] = 256 + int(extra)
    namespace["CODEWORDS_TO_ADD"] = int(extra)


def _value_span(namespace, count: int) -> tuple[int, int]:
    modulo = int(namespace["FILE_MODULO"])
    low = 10**9
    high = -10**9
    for index in range(count):
        grid = torch.load(
            f"transcribed_data_og/{index % modulo}/{index}.dat",
            weights_only=False,
        ).long()
        low = min(low, int(grid.min()))
        high = max(high, int(grid.max()))
    return low, high


def _summary(namespace, lengths, extra_rules, elapsed, split: str, missed: int | None) -> dict:
    baseline = int(namespace["WIDTH"] * namespace["HEIGHT"])
    lossless_bad, lossless_n = _lossless(namespace, len(lengths))
    low, high = _value_span(namespace, len(lengths))
    mean_length = sum(lengths) / len(lengths)
    payload = {
        "split": split,
        "images": len(lengths),
        "indices": list(namespace["CHOSEN_INDICES"]),
        "width": int(namespace["WIDTH"]),
        "height": int(namespace["HEIGHT"]),
        "baseline_tokens": baseline,
        "value_min": low,
        "value_max": high,
        "mean_length": mean_length,
        "compression_ratio": mean_length / baseline,
        "compression_percent_of_original": mean_length / baseline * 100.0,
        "lossless_bad": lossless_bad,
        "lossless_checked": lossless_n,
        "seconds": elapsed,
    }
    if split == "train":
        payload["rules"] = extra_rules
        payload["start_of_new_tokens"] = int(namespace["START_OF_NEW_TOKENS"])
        payload["extra_tokens"] = int(namespace["CODEWORDS_TO_ADD"])
    else:
        payload["extra_tokens"] = extra_rules
        payload["rules_missed_on_this_split"] = missed
    return payload


def _learn(work: Path, kind: str, per_class: int, offset: int, length: int, width: int, height: int, extra: int, channel) -> dict:
    os.chdir(REPO)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    os.chdir(work)
    started = time.time()
    cells = _cells()
    namespace = _namespace()
    _transcribe(namespace, cells, kind, "train", per_class, offset, length, width, height, extra, channel)
    print(
        f"learn {kind} channel={channel} images={len(namespace['train_loader'].dataset)} extra={extra}",
        flush=True,
    )
    exec(compile(cells[10], f"{NOTEBOOK}::cell10", "exec"), namespace, namespace)
    for index in (11, 12, 13, 14, 15):
        exec(compile(cells[index], f"{NOTEBOOK}::cell{index}", "exec"), namespace, namespace)
    resolve_source = cells[16].replace(
        "show(generate_sequence(WIDTH, HEIGHT)[None].float() / 255.0)",
        "pass",
    )
    exec(compile(resolve_source, f"{NOTEBOOK}::cell16", "exec"), namespace, namespace)
    lengths = [int(item) for item in namespace["sequence_lengths"]]
    payload = _summary(namespace, lengths, len(namespace["rules"]), time.time() - started, "train", None)
    (work / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("TRAIN", json.dumps({k: payload[k] for k in payload if k != "indices"}), flush=True)
    return payload


def _apply_test(work: Path, rules_dir: Path, kind: str, per_class: int, offset: int, length: int, width: int, height: int, extra: int, channel) -> dict:
    os.chdir(REPO)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    os.chdir(work)
    started = time.time()
    cells = _cells()
    namespace = _namespace()
    _transcribe(namespace, cells, kind, "test", per_class, offset, length, width, height, extra, channel)
    anchor_source = cells[10]
    anchor_end = anchor_source.index("\nCODEWORDS_ADDED")
    exec(compile(anchor_source[:anchor_end], f"{NOTEBOOK}::get_anchor", "exec"), namespace, namespace)
    rules = torch.load(rules_dir / "rules.dat", weights_only=False)
    train_meta = json.loads((rules_dir / "summary.json").read_text(encoding="utf-8"))
    start_of_new = int(train_meta["start_of_new_tokens"])
    namespace["START_OF_NEW_TOKENS"] = start_of_new
    replace_program = compile(_replace_block(cells[10]), f"{NOTEBOOK}::replace", "exec")
    missed = 0
    for index, rule in enumerate(rules):
        namespace["max_key"] = rule
        namespace["NEW_TOKEN_ID"] = start_of_new + index
        namespace["unique_train"] = namespace["UniqueDataset"]()
        namespace["last_output"] = time.time()
        exec(replace_program, namespace, namespace)
        if int(namespace["total_saved"]) <= 0:
            missed += 1
        if index == 0 or (index + 1) % 32 == 0 or index + 1 == len(rules):
            print(f"applied {index + 1}/{len(rules)} missed={missed}", flush=True)
    for index in (11, 12, 13, 14, 15):
        exec(compile(cells[index], f"{NOTEBOOK}::cell{index}", "exec"), namespace, namespace)
    resolve_source = cells[16].replace(
        "show(generate_sequence(WIDTH, HEIGHT)[None].float() / 255.0)",
        "pass",
    )
    exec(compile(resolve_source, f"{NOTEBOOK}::cell16", "exec"), namespace, namespace)
    lengths = [int(item) for item in namespace["sequence_lengths"]]
    payload = _summary(namespace, lengths, len(rules), time.time() - started, "test", missed)
    (work / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("TEST", json.dumps({k: payload[k] for k in payload if k != "indices"}), flush=True)
    return payload


def _usable(path: Path, images: int, extra: int) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("images", 0)) != int(images):
        return None
    if int(payload.get("extra_tokens", -1)) != int(extra):
        return None
    if payload.get("split") == "train" and int(payload.get("rules", 0)) != int(extra):
        return None
    return payload


def run_job(kind: str, setting: str, per_class: int, test_per_class: int, celeba_train: int, celeba_test: int, width: int, height: int, extra: int) -> dict:
    channel = CHANNELS[setting]
    root = RESULTS / kind / setting
    train_dir = root / "train"
    test_dir = root / "test"
    train_images = celeba_train if kind == "celeba" else per_class * 10
    test_images = celeba_test if kind == "celeba" else test_per_class * 10
    saved = _usable(train_dir / "summary.json", train_images, extra)
    if saved is not None and (train_dir / "rules.dat").exists():
        print("reuse train", kind, setting, saved.get("compression_ratio"), flush=True)
        train_summary = saved
    else:
        train_summary = _learn(
            train_dir,
            kind,
            per_class,
            0,
            celeba_train,
            width,
            height,
            extra,
            channel,
        )
    test_summary = _usable(test_dir / "summary.json", test_images, extra)
    if test_summary is None:
        test_summary = _apply_test(
            test_dir,
            train_dir,
            kind,
            test_per_class,
            celeba_train,
            celeba_test,
            width,
            height,
            extra,
            channel,
        )
    else:
        print("reuse test", kind, setting, test_summary.get("compression_ratio"), flush=True)
    return {"train": train_summary, "test": test_summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cifar", "svhn", "celeba"])
    parser.add_argument("--settings", nargs="+", default=["gray", "r", "g", "b"])
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--test-per-class", type=int, default=1)
    parser.add_argument("--celeba-train", type=int, default=10)
    parser.add_argument("--celeba-test", type=int, default=10)
    parser.add_argument("--extra", type=int, default=256)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--budget-seconds", type=int, default=105 * 60)
    args = parser.parse_args()

    print(f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)
    print(f"notebook_sha256={hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = {"jobs": [], "stopped_early": False}
    for kind in args.datasets:
        for setting in args.settings:
            elapsed = time.time() - started
            if elapsed > args.budget_seconds:
                report["stopped_early"] = True
                print(f"STOP before {kind} {setting} elapsed={elapsed:.1f}", flush=True)
                break
            print(f"JOB {kind} {setting}", flush=True)
            try:
                outcome = run_job(
                    kind,
                    setting,
                    args.per_class,
                    args.test_per_class,
                    args.celeba_train,
                    args.celeba_test,
                    args.width,
                    args.height,
                    args.extra,
                )
                report["jobs"].append({"dataset": kind, "setting": setting, "ok": True, **outcome})
            except Exception as exc:
                report["jobs"].append({"dataset": kind, "setting": setting, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
                print("JOB_ERROR", kind, setting, report["jobs"][-1]["error"], flush=True)
                report["stopped_early"] = True
                break
        if report["stopped_early"]:
            break
    report["elapsed_seconds"] = time.time() - started
    out = RESULTS / "summary.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("WROTE", out, flush=True)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
