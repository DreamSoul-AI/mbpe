"""Feed SVHN and CelebA through the official Python tokenizer.

The notebook on disk is not modified. Settings are substituted in a memory
copy: image size, how many extra symbols, and which images are loaded.
The merge loop is the notebook's own source.

Official transcription keeps channel 0 after ToTensor, multiplied by 255.
For RGB images that channel is red, stored as an integer grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
import http.cookiejar
import zipfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"

import torch

torch.set_num_threads(8)
try:
    torch.set_num_interop_threads(8)
except RuntimeError:
    pass

import matplotlib

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "third_party" / "MDBPE_TF" / "MDBPE" / "processor_python.ipynb"
DATA = REPO / "data"
RESULTS = REPO / "results" / "2026-11-10-elsner-svhn-celeba"

CELEBA_FILES = [
    ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
    ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
    ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
    ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
    ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
    ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
]
MAX_ITEM_BYTES = 200 * 1024**3
MIN_FREE_BYTES = 100 * 1024**3


def _cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cells.append("".join(cell.get("source", [])))
    return cells


def _dedent(text: str) -> str:
    lines = text.splitlines()
    indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
    cut = min(indents) if indents else 0
    return "\n".join(line[cut:] if len(line) >= cut else line for line in lines)


def _replace_block(cell: str) -> str:
    start = cell.index("    it = 0\n    total_saved = 0\n    for unique, classes, path in unique_train:")
    end = cell.index("    #increment token ID")
    return _dedent(cell[start:end])


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _free_bytes() -> int:
    return shutil.disk_usage("D:\\").free


def _size_text_to_bytes(text: str) -> int | None:
    raw = text.strip().upper().replace(" ", "")
    if not raw:
        return None
    factor = 1
    if raw.endswith("G"):
        factor = 1024**3
        raw = raw[:-1]
    elif raw.endswith("M"):
        factor = 1024**2
        raw = raw[:-1]
    elif raw.endswith("K"):
        factor = 1024
        raw = raw[:-1]
    try:
        return int(float(raw) * factor)
    except ValueError:
        return None


def _gdrive_download(file_id: str, dest: Path, expected_md5: str) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and _md5(dest) == expected_md5:
        return {"path": str(dest), "bytes": dest.stat().st_size, "cached": True}

    direct = (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )
    probe = urllib.request.Request(
        direct,
        headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"},
    )
    with urllib.request.urlopen(probe, timeout=60) as probed:
        span = probed.headers.get("Content-Range", "")
        kind = probed.headers.get("Content-Type", "")
    if "text/html" in kind or "/" not in span:
        raise RuntimeError(f"{dest.name} size was not visible before download ({kind}, {span})")
    exact = int(span.rsplit("/", 1)[1])
    if exact > MAX_ITEM_BYTES:
        raise RuntimeError(f"{dest.name} is {exact} bytes, over 200GB")
    if _free_bytes() - exact < MIN_FREE_BYTES:
        raise RuntimeError(f"{dest.name} would leave D: below 100GB")
    print(f"download {dest.name} bytes={exact}", flush=True)
    # A full GET is the quota warning page. Small byte ranges still return the zip.
    piece = 64 * 1024
    start = dest.stat().st_size if dest.exists() else 0
    if start > exact:
        dest.unlink()
        start = 0
    print(f"resume {dest.name} at {start}", flush=True)
    with dest.open("ab" if start else "wb") as handle:
        while start < exact:
            end = min(exact - 1, start + piece - 1)
            payload = b""
            for _attempt in range(3):
                request = urllib.request.Request(
                    direct,
                    headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"},
                )
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        kind = response.headers.get("Content-Type", "")
                        if "text/html" in kind:
                            raise RuntimeError("html")
                        payload = response.read()
                    if payload:
                        break
                except Exception:
                    payload = b""
                time.sleep(1)
            if not payload:
                if piece > 32 * 1024:
                    piece //= 2
                    print(f"shrink {dest.name} piece={piece}", flush=True)
                    continue
                raise RuntimeError(f"{dest.name} range {start} failed")
            handle.write(payload)
            start += len(payload)
            time.sleep(0.05)
            if start == exact or start % (32 * 1024 * 1024) < len(payload):
                print(f"  {dest.name} {start}/{exact}", flush=True)

    digest = _md5(dest)
    if digest != expected_md5:
        raise RuntimeError(f"{dest.name} md5 {digest} != {expected_md5}")
    return {"path": str(dest), "bytes": dest.stat().st_size, "page_size": exact, "cached": False}


def download_svhn() -> dict:
    from torchvision import datasets

    train = datasets.SVHN(str(DATA), split="train", download=True)
    test = datasets.SVHN(str(DATA), split="test", download=True)
    return {"train": len(train), "test": len(test)}


def download_celeba() -> dict:
    folder = DATA / "celeba"
    folder.mkdir(parents=True, exist_ok=True)
    report = []
    for file_id, digest, name in CELEBA_FILES:
        print(f"celeba file {name}", flush=True)
        report.append(_gdrive_download(file_id, folder / name, digest))
    image_dir = folder / "img_align_celeba"
    if not image_dir.is_dir():
        print("extracting img_align_celeba.zip", flush=True)
        with zipfile.ZipFile(folder / "img_align_celeba.zip") as archive:
            archive.extractall(folder)
    return {"files": report, "image_dir": image_dir.is_dir()}


PREPARED_GRIDS = None


def _subset_code(kind: str, split: str, per_class: int, count: int) -> str:
    root = str(DATA)
    if kind == "grids":
        return """
import PIL
class _GridImages(torch.utils.data.Dataset):
    def __init__(self, grids, transform):
        self.grids = grids
        self.transform = transform
    def __len__(self):
        return len(self.grids)
    def __getitem__(self, index):
        image = PIL.Image.fromarray(self.grids[index], mode="L")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0
mnist_trainset = _GridImages(PREPARED_GRIDS, transform)
train_loader = DataLoader(mnist_trainset, batch_size=8, shuffle=False, num_workers=0)
test_loader = train_loader
"""
    if kind == "svhn":
        return f"""
mnist_trainset = datasets.SVHN(root=r'{root}', split='{split}', download=False, transform=transform)
_labels = mnist_trainset.labels.reshape(-1)
_counts = {{}}
_idx = []
for _i, _y in enumerate(_labels.tolist()):
    _y = int(_y)
    if _counts.get(_y, 0) < {int(per_class)}:
        _idx.append(_i)
        _counts[_y] = _counts.get(_y, 0) + 1
mnist_trainset = torch.utils.data.Subset(mnist_trainset, _idx)
train_loader = DataLoader(mnist_trainset, batch_size=8, shuffle=False, num_workers=0)
test_loader = train_loader
"""
    # Official Drive quota stopped the zip tail and the split file.
    # These JPEGs are the first complete files inside the partial zip, in name order.
    folder = DATA / "celeba" / "partial_jpg"
    start = 0 if split == "train" else int(count)
    stop = start + int(count)
    return f"""
import PIL
from pathlib import Path
class _FolderImages(torch.utils.data.Dataset):
    def __init__(self, files, transform):
        self.files = list(files)
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        image = PIL.Image.open(self.files[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0
_files = sorted(Path(r'{folder}').glob('*.jpg'))[{start}:{stop}]
mnist_trainset = _FolderImages(_files, transform)
train_loader = DataLoader(mnist_trainset, batch_size=8, shuffle=False, num_workers=0)
test_loader = train_loader
"""


def _prepare_setup(source: str, width: int, height: int, extra: int) -> str:
    source = source.replace("WIDTH, HEIGHT = 12, 12", f"WIDTH, HEIGHT = {width}, {height}", 1)
    source = source.replace(
        "MAXIMUM_NUMBER_OF_TOKENS = 256+32 #256 = only use the base tokens, 256+32 --> use 32 extra tokens to compact stuff into",
        f"MAXIMUM_NUMBER_OF_TOKENS = 256+{int(extra)} #256 = only use the base tokens, 256+32 --> use 32 extra tokens to compact stuff into",
        1,
    )
    source = source.replace(
        'LIMITED_EXAMPLES = 100 #--> only use a subset of the data (for debugging); set to "None" to use all data',
        'LIMITED_EXAMPLES = None #--> only use a subset of the data (for debugging); set to "None" to use all data',
        1,
    )
    return source


def _namespace():
    namespace = {"__name__": "__main__"}
    exec(compile(_cells()[0], f"{NOTEBOOK}::cell0", "exec"), namespace, namespace)
    namespace["plt"].show = lambda *args, **kwargs: None
    return namespace


def _transcribe(namespace, cells, kind, split, per_class, count, width, height, extra):
    if kind == "grids":
        namespace["PREPARED_GRIDS"] = PREPARED_GRIDS
    exec(compile(_prepare_setup(cells[1], width, height, extra), f"{NOTEBOOK}::cell1", "exec"), namespace, namespace)
    dataset_cell = cells[2]
    dataset_cell = dataset_cell.replace(
        "mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)\n"
        "mnist_testnset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)\n"
        "train_loader = DataLoader(mnist_trainset, batch_size=8, shuffle=False)\n"
        "test_loader = DataLoader(mnist_testnset, batch_size=8, shuffle=False)\n",
        _subset_code(kind, split, per_class, count),
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


def _learn(work: Path, kind: str, per_class: int, count: int, width: int, height: int, extra: int) -> dict:
    os.chdir(REPO)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    os.chdir(work)
    started = time.time()
    cells = _cells()
    namespace = _namespace()
    _transcribe(namespace, cells, kind, "train", per_class, count, width, height, extra)
    print(
        f"learn {kind} images={len(namespace['train_loader'].dataset)} "
        f"start={namespace['START_OF_NEW_TOKENS']} extra={extra}",
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
    baseline = int(namespace["WIDTH"] * namespace["HEIGHT"])
    lossless_bad, lossless_n = _lossless(namespace, len(lengths))
    elapsed = time.time() - started
    summary = {
        "split": "train",
        "images": len(lengths),
        "width": int(namespace["WIDTH"]),
        "height": int(namespace["HEIGHT"]),
        "baseline_tokens": baseline,
        "extra_tokens": int(extra),
        "rules": len(namespace["rules"]),
        "start_of_new_tokens": int(namespace["START_OF_NEW_TOKENS"]),
        "mean_length": sum(lengths) / len(lengths),
        "compression_percent_of_original": sum(lengths) / len(lengths) / baseline * 100.0,
        "lossless_bad": lossless_bad,
        "lossless_checked": lossless_n,
        "seconds": elapsed,
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("TRAIN", json.dumps(summary), flush=True)
    return summary


def _apply_test(work: Path, rules_dir: Path, kind: str, per_class: int, count: int, width: int, height: int, extra: int) -> dict:
    os.chdir(REPO)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    os.chdir(work)
    started = time.time()
    cells = _cells()
    namespace = _namespace()
    _transcribe(namespace, cells, kind, "test", per_class, count, width, height, extra)
    anchor_source = cells[10]
    anchor_end = anchor_source.index("\nCODEWORDS_ADDED")
    exec(compile(anchor_source[:anchor_end], f"{NOTEBOOK}::get_anchor", "exec"), namespace, namespace)
    rules = torch.load(rules_dir / "rules.dat", weights_only=False)
    train_meta = json.loads((rules_dir / "summary.json").read_text(encoding="utf-8"))
    start_of_new = int(train_meta["start_of_new_tokens"])
    namespace["START_OF_NEW_TOKENS"] = start_of_new
    replace_code = _replace_block(cells[10])
    missed = 0
    for index, rule in enumerate(rules):
        namespace["max_key"] = rule
        namespace["NEW_TOKEN_ID"] = start_of_new + index
        namespace["unique_train"] = namespace["UniqueDataset"]()
        namespace["last_output"] = time.time()
        exec(compile(replace_code, f"{NOTEBOOK}::replace", "exec"), namespace, namespace)
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
    baseline = int(namespace["WIDTH"] * namespace["HEIGHT"])
    lossless_bad, lossless_n = _lossless(namespace, len(lengths))
    elapsed = time.time() - started
    summary = {
        "split": "test",
        "images": len(lengths),
        "width": int(namespace["WIDTH"]),
        "height": int(namespace["HEIGHT"]),
        "baseline_tokens": baseline,
        "extra_tokens": len(rules),
        "rules_missed_on_this_split": missed,
        "mean_length": sum(lengths) / len(lengths),
        "compression_percent_of_original": sum(lengths) / len(lengths) / baseline * 100.0,
        "lossless_bad": lossless_bad,
        "lossless_checked": lossless_n,
        "seconds": elapsed,
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("TEST", json.dumps(summary), flush=True)
    return summary


def _lossless(namespace, count: int) -> tuple[int, int]:
    bad = 0
    modulo = int(namespace["FILE_MODULO"])
    width = int(namespace["WIDTH"])
    height = int(namespace["HEIGHT"])
    for index in range(count):
        sequence = torch.load(
            f"tokensequences/{index % modulo}/{index}.dat",
            weights_only=False,
        )
        grid = torch.ones(width, height, dtype=torch.long) * -1
        try:
            for token, x_pos, y_pos in sequence:
                grid = namespace["resolve_token"](grid, int(token), int(x_pos), int(y_pos))
            original = torch.load(
                f"transcribed_data_og/{index % modulo}/{index}.dat",
                weights_only=False,
            ).long()
            if grid.shape != original.shape or not torch.equal(grid, original):
                bad += 1
        except Exception:
            bad += 1
    return bad, count


def run_dataset(kind: str, per_class: int, count: int, width: int, height: int, extra: int, test: bool) -> dict:
    root = RESULTS / kind
    root.mkdir(parents=True, exist_ok=True)
    train_dir = root / "train"
    saved = train_dir / "summary.json"
    if saved.exists() and (train_dir / "rules.dat").exists():
        train_summary = json.loads(saved.read_text(encoding="utf-8"))
        if int(train_summary.get("rules", 0)) != int(extra) or int(train_summary.get("images", 0)) <= 0:
            train_summary = _learn(train_dir, kind, per_class, count, width, height, extra)
        else:
            print("reuse train", kind, train_summary.get("mean_length"), flush=True)
    else:
        train_summary = _learn(train_dir, kind, per_class, count, width, height, extra)
    test_summary = None
    if test and extra > 0:
        test_summary = _apply_test(root / "test", root / "train", kind, per_class, count, width, height, extra)
    return {"train": train_summary, "test": test_summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("svhn", "celeba", "both"), default="both")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--celeba-count", type=int, default=100)
    parser.add_argument("--extra", type=int, default=256)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args()

    print(f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)
    print(f"free_bytes={_free_bytes()}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"free_bytes_before": _free_bytes()}
    if args.dataset in ("svhn", "both"):
        if not args.skip_download:
            report["svhn_download"] = download_svhn()
            print("SVHN", report["svhn_download"], flush=True)
        report["svhn"] = run_dataset(
            "svhn", args.per_class, args.celeba_count, args.width, args.height, args.extra, not args.no_test
        )
    if args.dataset in ("celeba", "both"):
        try:
            if not args.skip_download:
                report["celeba_download"] = download_celeba()
                print("CELEBA_DOWNLOAD_OK", flush=True)
            report["celeba"] = run_dataset(
                "celeba", args.per_class, args.celeba_count, args.width, args.height, args.extra, not args.no_test
            )
        except Exception as exc:
            report["celeba_error"] = f"{type(exc).__name__}: {exc}"
            print("CELEBA_ERROR", report["celeba_error"], flush=True)
    out = RESULTS / "summary.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("WROTE", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
