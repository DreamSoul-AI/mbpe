"""Compare complete BPE files on unscaled CelebA crops using a frozen split."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import zlib

from batch30_mask_baselines import (
    alphabet_of, cells_2d, flatten, learn_apply, mask_image, merge_one, pick, slice_model,
)
from batch31_mnist import classify
from bpe_block_file import FileSpec, ORDERS, VERSION, ZLIB_LEVEL, orders_for, write_archive
from lossless_nd_candidate import LosslessNDCandidate
import numpy as np
from PIL import Image, __version__ as pillow_version
import psutil


SIDE = 128
SEED = 20261113
BUDGETS = (0, 64, 256)
SPLIT_SIZES = {"learn": 24, "validation": 16, "test": 32}
MASKS = ("4level", "bin128")
STUDIES = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_save(path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def freeze_inputs(source):
    unique, rejected = [], []
    seen = {mask: set() for mask in MASKS}
    crops = {}
    for path in sorted(source.glob("*.jpg")):
        with Image.open(path) as image:
            width, height = image.size
            if min(width, height) < SIDE:
                rejected.append({"file": path.name, "reason": "image smaller than crop"})
                continue
            left, top = (width - SIDE) // 2, (height - SIDE) // 2
            box = (left, top, left + SIDE, top + SIDE)
            gray = np.asarray(image.crop(box).convert("L"), dtype=np.uint8)
        masked = {mask: mask_image(gray, mask).astype(np.uint8) for mask in MASKS}
        digests = {mask: sha256(grid.tobytes()) for mask, grid in masked.items()}
        if any(digests[mask] in seen[mask] for mask in MASKS):
            rejected.append({"file": path.name, "reason": "duplicate cropped grid in at least one mask"})
            continue
        for mask, digest in digests.items():
            seen[mask].add(digest)
        crops[path.name] = masked
        unique.append({
            "file": path.name, "source_sha256": sha256(path.read_bytes()),
            "size": (width, height), "crop": box, "gray_sha256": sha256(gray.tobytes()),
            "grid_sha256": digests,
            "value_counts": {mask: np.bincount(grid.ravel(), minlength=alphabet_of(mask)).tolist()
                             for mask, grid in masked.items()},
        })
    if len(unique) < sum(SPLIT_SIZES.values()):
        raise ValueError(f"only {len(unique)} distinct crops available")
    random.Random(SEED).shuffle(unique)
    splits, start = {}, 0
    for name, size in SPLIT_SIZES.items():
        splits[name] = unique[start:start + size]
        start += size
    return crops, {
        "source": str(source.resolve()), "seed": SEED, "side": SIDE, "source_count": len(unique) + len(rejected),
        "grouping": "source filename, exact duplicate grids excluded, identity metadata unavailable",
        "splits": splits, "excluded": rejected, "unused": unique[start:],
    }


def candidate_spec(model, alphabet):
    if model.base != list(range(alphabet)):
        raise ValueError("learning split did not cover the experiment's alphabet")
    rules = tuple((left, right, *delta) for (left, right, delta), _ in model.rules)
    return FileSpec("2d", SIDE, alphabet, rules)


def shape_replacements(model, sequences):
    # Every merge remains as an internal node of a surviving token's expansion.
    uses = {symbol: Counter() for symbol in range(len(model.base))}
    for (left, right, _), token in model.rules:
        counts = uses[left] + uses[right]
        counts[classify(model.shapes[token])[0]] += 1
        uses[token] = counts
    observed = Counter()
    for sequence in sequences:
        for token, count in Counter(sequence).items():
            for kind, occurrences in uses[token].items():
                observed[kind] += count * occurrences
    if sum(observed.values()) != sum(SIDE * SIDE - len(sequence) for sequence in sequences):
        raise ValueError("merge count does not agree with token count")
    return {kind: observed[kind] for kind in ("strip", "compact", "other")}


def candidate_sequences(model, images):
    sequences = []
    for image in images:
        packet = model.encode((SIDE, SIDE), cells_2d(image))
        if packet["domain"] is not None or packet["literals"]:
            raise ValueError("expected a dense grid with no out-of-alphabet values")
        sequences.append(packet["tokens"])
    return sequences


def linear_sequences(spec, images):
    order = orders_for(SIDE)[spec.mode]
    sequences = []
    for image in images:
        sequence = flatten(image, order)
        for token, (left, right) in enumerate(spec.rules, start=spec.alphabet):
            sequence = merge_one(sequence, left, right, token)
        sequences.append(sequence.tolist())
    return sequences


def compare_grids(decoded, originals):
    restored = np.frombuffer(decoded, dtype=np.uint8)
    if restored.size != len(originals) * SIDE * SIDE:
        raise ValueError(f"decoded length mismatch: {restored.size} bytes")
    failures = []
    for i, (image, expected) in enumerate(zip(restored.reshape(-1, SIDE, SIDE), originals)):
        differing = np.flatnonzero(image != expected)
        if differing.size:
            y, x = divmod(int(differing[0]), SIDE)
            failures.append({"image_index": i, "mismatched_cells": int(differing.size),
                             "first_cell": [y, x], "decoded": int(image[y, x]), "expected": int(expected[y, x])})
    return failures


def measure(directory, name, spec, sequences, originals):
    archive = directory / f"{name}.mbpe"
    restored = directory / f"{name}.decoded"
    started = time.perf_counter()
    with archive.open("xb") as handle:
        result = write_archive(handle, spec, sequences)
    result["write_seconds"] = time.perf_counter() - started
    result["file_sha256"] = sha256(archive.read_bytes())
    if archive.stat().st_size != result["total_bytes"]:
        raise ValueError("archive size does not agree with byte accounting")
    result["file"] = archive.name
    result["tokens"] = sum(result["token_counts"])
    result["bits_per_cell"] = 8 * result["total_bytes"] / (len(originals) * spec.cells)

    started = time.perf_counter()
    decoder = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", str(STUDIES / "bpe_block_file.py"), str(archive), str(restored)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    result["decode_seconds"] = time.perf_counter() - started
    if decoder.returncode:
        result["decoder_error"] = decoder.stderr
        return result
    readback = json.loads(decoder.stdout)
    result["decoder"] = {key: readback.pop(key) for key in ("peak_working_set_bytes", "cpu_affinity", "priority")}
    result["decoded_sha256"] = sha256(restored.read_bytes())
    result["lossless_bad"] = compare_grids(restored.read_bytes(), originals)
    if any(readback.get(key) != result[key] for key in readback):
        raise ValueError("decoded metadata does not agree with the writer")
    if result["decoder"]["priority"] != int(psutil.IDLE_PRIORITY_CLASS) or len(result["decoder"]["cpu_affinity"]) != 1:
        raise ValueError("decoder did not inherit the run's CPU constraints")
    return result


def remember(directory, name, rows, spec, sequences, images, model=None):
    result = measure(directory, name, spec, sequences, images)
    if model is not None:
        result["shape_replacements"] = shape_replacements(model, sequences)
    rows[name] = result
    json_save(directory / f"{name}.json", result)
    if result.get("decoder_error") or result["lossless_bad"]:
        raise ValueError(f"round-trip failed for {directory.name}/{name}")
    print(f"{directory.parent.name}/{directory.name} {name}: "
          f"{result['total_bytes']} bytes, {result['tokens']} tokens, all grids restored", flush=True)


def compare_files(candidate, baseline):
    diff = np.asarray(candidate["payload_sizes"], dtype=np.int64) - baseline["payload_sizes"]
    fixed = candidate["total_bytes"] - baseline["total_bytes"] - int(diff.sum())
    indices = np.random.default_rng(SEED).integers(0, len(diff), size=(10_000, len(diff)))
    low, high = np.quantile(fixed + diff[indices].sum(axis=1), [0.025, 0.975])
    return {
        "total_delta_bytes": candidate["total_bytes"] - baseline["total_bytes"],
        "fixed_delta_bytes": fixed,
        "payload_delta_bytes": int(diff.sum()),
        "bootstrap_95_delta_bytes": [float(low), float(high)],
        "payload_win_tie_loss": [int((diff < 0).sum()), int((diff == 0).sum()), int((diff > 0).sum())],
    }


def run_mask(directory, mask, crops, manifest):
    directory.mkdir()
    validation_dir, test_dir = directory / "validation", directory / "test"
    validation_dir.mkdir()
    test_dir.mkdir()
    alphabet = alphabet_of(mask)
    learn = [crops[item["file"]][mask] for item in manifest["splits"]["learn"]]
    validation = [crops[item["file"]][mask] for item in manifest["splits"]["validation"]]
    model = LosslessNDCandidate(2)
    started = time.perf_counter()
    print(f"{mask} learning 2D, available={psutil.virtual_memory().available}", flush=True)
    model.learn([((SIDE, SIDE), cells_2d(image)) for image in learn], budget=max(BUDGETS), score="frequency")
    learning_seconds = {"2d": time.perf_counter() - started}
    json_save(directory / "2d-dictionary.json", model.dictionary())

    validation_rows, specs = {}, {}
    for budget in BUDGETS:
        part = slice_model(model, budget)
        name = f"2d-{budget}"
        specs[name] = candidate_spec(part, alphabet)
        sequences = candidate_sequences(part, validation)
        remember(validation_dir, name, validation_rows, specs[name], sequences, validation, part)
    for mode in ORDERS:
        order = orders_for(SIDE)[mode]
        valid_sequences = [flatten(image, order) for image in validation]
        started = time.perf_counter()
        rules, snapshots = learn_apply(
            [flatten(image, order) for image in learn], {"validation": valid_sequences},
            alphabet, max(BUDGETS), set(BUDGETS[1:]),
        )
        learning_seconds[mode] = time.perf_counter() - started
        json_save(directory / f"{mode}-rules.json", rules)
        for budget in BUDGETS:
            if budget:
                (sequences, bad), used = pick(snapshots["validation"], budget)
                if bad:
                    raise ValueError(f"linear expansion failed before serialization: {mode}-{budget}")
            else:
                used, sequences = 0, [values.tolist() for values in valid_sequences]
            name = f"{mode}-{budget}"
            specs[name] = FileSpec(mode, SIDE, alphabet, tuple((left, right) for left, right, _ in rules[:used]))
            remember(validation_dir, name, validation_rows, specs[name], sequences, validation)
    raw_spec = specs["raw"] = FileSpec("raw", SIDE, alphabet)
    remember(validation_dir, "raw", validation_rows, raw_spec, [grid.ravel() for grid in validation], validation)

    def best(modes, *, learned_only=False):
        return min(
            (name for name, spec in specs.items() if spec.mode in modes and (spec.rules or not learned_only)),
            key=lambda name: (validation_rows[name]["total_bytes"], len(specs[name].rules), name),
        )

    selected = {"2d": best(("2d",)), "best_1d": best(ORDERS), "hilbert": best(("hilbert",)), "zlib": "raw"}
    learned = {
        "2d": best(("2d",), learned_only=True), "best_1d": best(ORDERS, learned_only=True),
        "hilbert": best(("hilbert",), learned_only=True),
    }
    json_save(directory / "selected-before-test.json", {"primary": selected, "learned_only": learned})
    print(f"{mask} validation selections: {selected}, learned only: {learned}", flush=True)

    test = [crops[item["file"]][mask] for item in manifest["splits"]["test"]]
    test_rows = {}
    for name in dict.fromkeys((*selected.values(), *learned.values())):
        spec = specs[name]
        part = None
        if spec.mode == "2d":
            part = slice_model(model, len(spec.rules))
            sequences = candidate_sequences(part, test)
        elif spec.mode == "raw":
            sequences = [grid.ravel() for grid in test]
        else:
            sequences = linear_sequences(spec, test)
        remember(test_dir, name, test_rows, spec, sequences, test, part)
    candidate = test_rows[selected["2d"]]
    learned_candidate = test_rows[learned["2d"]]
    return {
        "selected": selected, "selected_learned_only": learned, "learning_and_validation_apply_seconds": learning_seconds,
        "validation": validation_rows, "test": test_rows,
        "comparison": {role: compare_files(candidate, test_rows[selected[role]])
                       for role in ("best_1d", "hilbert", "zlib")},
        "learned_comparison": {role: compare_files(learned_candidate, test_rows[learned[role]])
                               for role in ("best_1d", "hilbert")},
        "learned_vs_zlib": compare_files(learned_candidate, test_rows[selected["zlib"]]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:1])
    started = time.perf_counter()
    crops, manifest = freeze_inputs(args.source)
    json_save(args.out / "inputs.json", manifest)
    dependencies = (
        "batch33_celeba_files.py", "bpe_block_file.py", "batch31_mnist.py", "batch30_mask_baselines.py",
        "lossless_nd_candidate.py", "nd_bpe_incremental_probe.py", "nd_bpe_graph_probe.py", "nd_bpe_probe.py",
    )
    report = {
        "complete": False, "python": sys.version, "numpy": np.__version__, "pillow": pillow_version,
        "psutil": psutil.__version__, "zlib": zlib.ZLIB_RUNTIME_VERSION, "zlib_level": ZLIB_LEVEL,
        "file_version": VERSION, "cpu_affinity": process.cpu_affinity(), "priority": int(process.nice()),
        "data_workers": 0, "budget": BUDGETS, "seed": SEED,
        "available_memory_start_bytes": psutil.virtual_memory().available,
        "inputs_sha256": sha256((args.out / "inputs.json").read_bytes()),
        "source_sha256": {name: sha256((STUDIES / name).read_bytes()) for name in dependencies},
        "results": {},
    }
    json_save(args.out / "run-config.json", report)
    for mask in MASKS:
        report["results"][mask] = run_mask(args.out / mask, mask, crops, manifest)
        json_save(args.out / mask / "summary.json", report["results"][mask])
    memory = process.memory_info()
    report.update(complete=True, seconds=time.perf_counter() - started,
                  peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss))
    json_save(args.out / "report.json", report)
    print(f"Complete in {report['seconds']:.1f}s, peak working set {report['peak_working_set_bytes']} bytes", flush=True)


if __name__ == "__main__":
    main()
