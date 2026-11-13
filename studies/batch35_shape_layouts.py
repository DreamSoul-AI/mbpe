"""Compare paired compact-block and stripe grids with equal BPE budgets."""

import argparse
import json
from pathlib import Path
import sys
import time
import zlib

from batch30_mask_baselines import cells_2d, flatten, learn_apply, pick, slice_model
from batch33_celeba_files import SIDE, STUDIES, candidate_sequences, candidate_spec, json_save, sha256, shape_replacements
from batch34_token_coding import apply_prefix, write_checked
from bpe_block_file import FileSpec, orders_for, write_archive, write_huffman_archive
from canonical_huffman import lengths_from_validation
from lossless_nd_candidate import LosslessNDCandidate
import numpy as np
import psutil


ALPHABET = 4
TILES = 7
TILE_SIDE = 16
SPLITS = {"learn": (24, 35001), "validation": (16, 35002), "test": (32, 35003)}
LAYOUTS = ("compact", "stripe")
BUDGETS = (0, 64, 256)
BOOTSTRAP_SEED = 35004


def make_pairs(count, seed):
    rng = np.random.default_rng(seed)
    filled_side = TILES * TILE_SIDE
    label_pool = np.repeat(np.arange(ALPHABET, dtype=np.uint8), 4)
    for i in range(count):
        labels = np.asarray([rng.permutation(label_pool) for _ in range(TILES**2)])
        compact = labels.reshape(TILES, TILES, 4, 4).repeat(4, axis=2).repeat(4, axis=3)
        stripe = labels.reshape(TILES, TILES, TILE_SIDE, 1).repeat(TILE_SIDE, axis=3)
        y, x = map(int, rng.integers(0, SIDE - filled_side + 1, size=2))
        grids = {}
        for name, tiles in zip(LAYOUTS, (compact, stripe)):
            grid = np.zeros((SIDE, SIDE), dtype=np.uint8)
            grid[y:y + filled_side, x:x + filled_side] = tiles.transpose(0, 2, 1, 3).reshape(filled_side, filled_side)
            grids[name] = grid.T.copy() if i % 2 else grid
        expected = np.full(ALPHABET, filled_side**2 // ALPHABET)
        expected[0] += SIDE**2 - filled_side**2
        if any(not np.array_equal(np.bincount(grid.ravel(), minlength=ALPHABET), expected)
               for grid in grids.values()):
            raise ValueError("layout pair does not have the expected value counts")
        yield grids, {
            "index": i, "offset_yx_before_transpose": [y, x], "transposed": bool(i % 2),
            "labels": labels.tolist(), "value_counts": expected.tolist(),
            "grid_sha256": {name: sha256(grid.tobytes()) for name, grid in grids.items()},
        }


def freeze_pairs():
    data, rows = {}, {}
    seen = set()
    for split, (count, seed) in SPLITS.items():
        data[split] = {name: [] for name in LAYOUTS}
        rows[split] = []
        for grids, record in make_pairs(count, seed):
            for name, grid in grids.items():
                digest = record["grid_sha256"][name]
                if digest in seen:
                    raise ValueError("duplicate grid across the frozen inputs")
                seen.add(digest)
                data[split][name].append(grid)
            rows[split].append(record)
    return data, {"splits": rows, "parameters": SPLITS, "side": SIDE, "alphabet": ALPHABET,
                  "tiles_per_axis": TILES, "tile_side": TILE_SIDE}


def record_files(output, name, spec, validation, test, reference, manifest, layout):
    lengths = lengths_from_validation(validation, spec.symbols)
    result = {}
    for codec, writer in (
        ("huffman", lambda handle: write_huffman_archive(handle, spec, test, lengths)),
        ("zlib", lambda handle: write_archive(handle, spec, test)),
    ):
        result[codec] = write_checked(
            output, f"{name}-{codec}", writer, spec, test, reference, manifest, layout, "test",
        )
    return result


def run_layout(output, layout, data, manifest):
    output.mkdir()
    learn, validation, test = (data[split][layout] for split in SPLITS)
    reference = b"".join(image.tobytes() for image in test)
    result = {"files": {}, "shape_replacements": {}, "seconds": {}}
    model = LosslessNDCandidate(2)
    start = time.perf_counter()
    print(f"{layout}: learning 2D", flush=True)
    model.learn([((SIDE, SIDE), cells_2d(image)) for image in learn], budget=max(BUDGETS), score="frequency")
    result["seconds"]["learn_2d"] = time.perf_counter() - start
    order = orders_for(SIDE)["hilbert"]
    validation_linear = [flatten(image, order) for image in validation]
    test_linear = [flatten(image, order) for image in test]
    start = time.perf_counter()
    rules, snapshots = learn_apply(
        [flatten(image, order) for image in learn], {"validation": validation_linear},
        ALPHABET, max(BUDGETS), set(BUDGETS[1:]),
    )
    result["seconds"]["learn_and_apply_validation_hilbert"] = time.perf_counter() - start
    if len(model.rules) != max(BUDGETS) or len(rules) != max(BUDGETS):
        raise ValueError(f"could not reach a matched rule budget for {layout}")
    json_save(output / "2d-dictionary.json", model.dictionary())
    json_save(output / "hilbert-rules.json", rules)
    for budget in BUDGETS:
        part = slice_model(model, budget)
        spec = candidate_spec(part, ALPHABET)
        start = time.perf_counter()
        if budget:
            valid_2d = candidate_sequences(part, validation)
            test_2d = candidate_sequences(part, test)
        else:
            valid_2d, test_2d = ([image.ravel().tolist() for image in images] for images in (validation, test))
        result["seconds"][f"apply_2d_{budget}"] = time.perf_counter() - start
        result["shape_replacements"][str(budget)] = shape_replacements(part, test_2d)
        result["files"][f"2d-{budget}"] = record_files(
            output, f"2d-{budget}", spec, valid_2d, test_2d, reference, manifest, layout,
        )

        spec = FileSpec("hilbert", SIDE, ALPHABET, tuple((left, right) for left, right, _ in rules[:budget]))
        if budget:
            (valid_1d, failures), used = pick(snapshots["validation"], budget)
            if failures or used != budget:
                raise ValueError("Hilbert validation failed or used a different budget")
        else:
            valid_1d = [sequence.tolist() for sequence in validation_linear]
        start = time.perf_counter()
        test_1d = apply_prefix(spec, test_linear)
        result["seconds"][f"apply_hilbert_{budget}"] = time.perf_counter() - start
        result["files"][f"hilbert-{budget}"] = record_files(
            output, f"hilbert-{budget}", spec, valid_1d, test_1d, reference, manifest, layout,
        )

    raw = [image.ravel().tolist() for image in test]
    spec = FileSpec("raw", SIDE, ALPHABET)
    result["files"]["raw"] = {"zlib": write_checked(
        output, "raw-zlib", lambda handle: write_archive(handle, spec, raw),
        spec, raw, reference, manifest, layout, "test",
    )}
    if result["files"]["raw"]["zlib"]["payload_sizes"] != result["files"]["2d-0"]["zlib"]["payload_sizes"]:
        raise ValueError("row-major zero-rule streams have different payloads")
    json_save(output / "summary.json", result)
    return result


def summarize_differences(results):
    count = SPLITS["test"][0]
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(count, size=(10_000, count))

    def summarize(values, fixed):
        return {
            "total": int(values.sum()) + fixed, "fixed": fixed,
            "per_pair": values.tolist(),
            "bootstrap_95": np.quantile(fixed + values[draws].sum(axis=1), (0.025, 0.975)).tolist(),
            "win_tie_loss": [int((values < 0).sum()), int((values == 0).sum()), int((values > 0).sum())],
        }

    comparison = {}
    for budget in BUDGETS:
        by_metric = comparison[str(budget)] = {}
        for metric in ("tokens", "huffman_bytes", "zlib_bytes"):
            diffs = {}
            for layout, result in results.items():
                files = result["files"]
                codec = "zlib" if metric == "zlib_bytes" else "huffman"
                nd, linear = (files[f"{method}-{budget}"][codec] for method in ("2d", "hilbert"))
                if metric == "tokens":
                    field, fixed = "token_counts", 0
                else:
                    field = "payload_sizes"
                    fixed = (nd["total_bytes"] - nd["payload_bytes"]) - (linear["total_bytes"] - linear["payload_bytes"])
                diffs[layout] = (np.asarray(nd[field], dtype=np.int64) - linear[field], fixed)
            compact, compact_fixed = diffs["compact"]
            stripe, stripe_fixed = diffs["stripe"]
            by_metric[metric] = {
                "compact": summarize(compact, compact_fixed),
                "stripe": summarize(stripe, stripe_fixed),
                "interaction": summarize(compact - stripe, compact_fixed - stripe_fixed),
            }
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:1])
    started = time.perf_counter()
    data, manifest = freeze_pairs()
    json_save(args.out / "inputs.json", manifest)
    dependencies = ("batch35_shape_layouts.py", "batch34_token_coding.py", "batch33_celeba_files.py",
                    "bpe_block_file.py", "canonical_huffman.py", "batch30_mask_baselines.py", "batch31_mnist.py",
                    "lossless_nd_candidate.py", "nd_bpe_incremental_probe.py", "nd_bpe_graph_probe.py", "nd_bpe_probe.py")
    report = {
        "complete": False, "python": sys.version, "numpy": np.__version__, "psutil": psutil.__version__,
        "zlib": zlib.ZLIB_RUNTIME_VERSION, "cpu_affinity": process.cpu_affinity(), "priority": int(process.nice()),
        "data_workers": 0, "budgets": BUDGETS, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_repetitions": 10_000,
        "available_memory_start_bytes": psutil.virtual_memory().available,
        "inputs_sha256": sha256((args.out / "inputs.json").read_bytes()),
        "source_sha256": {name: sha256((STUDIES / name).read_bytes()) for name in dependencies},
    }
    json_save(args.out / "run-config.json", report)
    results = {layout: run_layout(args.out / layout, layout, data, manifest) for layout in LAYOUTS}
    memory = process.memory_info()
    report.update(complete=True, results=results, comparison=summarize_differences(results),
                  seconds=time.perf_counter() - started,
                  peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss))
    json_save(args.out / "report.json", report)
    print(json.dumps({"seconds": report["seconds"], "peak_working_set_bytes": report["peak_working_set_bytes"]}),
          flush=True)


if __name__ == "__main__":
    main()
