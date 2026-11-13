"""Compare complete BPE files on original CIFAR-10 and SVHN green-channel grids."""

import argparse
from dataclasses import asdict
from pathlib import Path
import pickle
import sys
import time
import zlib

import grid_bpe_file as archive
from grid_bpe_study import (
    constrain_cpu, learn_linear, learn_nd, linear_packets, measure, nd_packets, nd_spec, reference_bytes, write_checked,
)
from batch30_mask_baselines import cells_2d
from batch33_celeba_files import json_save, sha256
from batch34_token_coding import read_json
from batch37_adaptive_scan import differences, neighbor_score
import numpy as np
import scipy
from scipy.io import loadmat


BUDGETS = (0, 64, 256)
SCANS = ("row", "column", "snake", "hilbert")
SEEDS = {"learn": 38001, "validation": 38002, "test": 38003}
PER_CLASS = {"learn": 10, "validation": 5, "test": 10}
SOURCES = {
    "cifar": ("cifar-10-batches-py/data_batch_5", "cifar-10-batches-py/test_batch"),
    "svhn": ("train_32x32.mat", "test_32x32.mat"),
}


def read_source(dataset, path, digest):
    if sha256(path.read_bytes()) != digest:
        raise ValueError(f"source does not match the local backup manifest: {path}")
    if dataset == "cifar":
        with path.open("rb") as handle:
            row = pickle.load(handle, encoding="bytes")
        images = row[b"data"].reshape(-1, 3, 32, 32)[:, 1]
        labels = np.asarray(row[b"labels"])
    else:
        row = loadmat(path, variable_names=("X", "y"))
        images = np.moveaxis(row["X"][:, :, 1, :], -1, 0)
        labels = row["y"].ravel() % 10
    if images.shape != (len(labels), 32, 32) or images.dtype != np.uint8 or set(labels) != set(range(10)):
        raise ValueError(f"unexpected source grid or label layout: {path}")
    return images, labels


def freeze_inputs(backup):
    entries = {entry["path"]: entry["sha256"] for entry in read_json(backup / "manifest.json")["data"]["entries"]}
    data, manifest = {}, {}
    for dataset, (train, test) in SOURCES.items():
        seen, picked, samples = set(), {}, {}
        for source, splits in ((train, ("learn", "validation")), (test, ("test",))):
            path = f"data/{source}"
            images, labels = read_source(dataset, backup / path, entries[path])
            for split in splits:
                rows, grids = [], []
                rng = np.random.default_rng(SEEDS[split])
                for label in range(10):
                    count = 0
                    for index in rng.permutation(np.flatnonzero(labels == label)):
                        grid = images[index]
                        digest = sha256(grid.tobytes())
                        if digest in seen:
                            continue
                        seen.add(digest)
                        rows.append({"index": int(index), "class": label, "sha256": digest})
                        grids.append(grid.copy())
                        count += 1
                        if count == PER_CLASS[split]:
                            break
                    if count != PER_CLASS[split]:
                        raise ValueError(f"not enough unique {dataset}/{split} grids for class {label}")
                picked[split] = {"source": path, "source_sha256": entries[path], "samples": rows}
                samples[split] = grids
            del images, labels
        data[dataset], manifest[dataset] = samples, picked
    return data, manifest


def as_samples(grids):
    return [(grid.shape, cells_2d(grid)) for grid in grids]


def fit_threshold(scores, row_sizes, column_sizes):
    candidates = sorted(set(scores) | {0, max(scores) + 1})
    trials = [(threshold, sum(row if score >= threshold else column
                             for score, row, column in zip(scores, row_sizes, column_sizes, strict=True)))
              for threshold in candidates]
    best, _ = min(trials, key=lambda item: (item[1], abs(item[0]), item[0]))
    return best, trials


def score_validation(spec, packets):
    directions = [0] * len(packets)
    table = archive.code_lengths(spec, directions, packets)
    costs = {codec: measure(spec, directions, packets, codec, table if codec == "huffman" else None)
             for codec in archive.CODECS}
    return spec, table, packets, costs


def prepare_models(grids, directory):
    learn, validation = (as_samples(grids[split]) for split in ("learn", "validation"))
    base = tuple(sorted({value for _, cells in learn for value in cells.values()}))
    linear = {mode: learn_linear(learn, base, mode, max(BUDGETS)) for mode in SCANS}
    nd = learn_nd(learn, max(BUDGETS))
    scores = [neighbor_score(grid) for grid in grids["validation"]]
    frozen = {}
    for budget in BUDGETS:
        choices = {}
        for mode in SCANS:
            spec = archive.Spec(mode, 2, base, (linear[mode][:budget],))
            choices[mode] = score_validation(spec, linear_packets(spec, validation))
        selected = {codec: min(SCANS, key=lambda mode: (
            choices[mode][3][codec]["total_bytes"], choices[mode][3][codec]["tokens"], mode,
        )) for codec in archive.CODECS}

        directional = []
        for mode in ("row", "column"):
            spec = archive.Spec(mode, 2, base, (linear[mode][:budget // 2],))
            directional.append(score_validation(spec, linear_packets(spec, validation)))
        row, column = directional
        threshold, trials = fit_threshold(scores, *(entry[3]["huffman"]["payload_sizes"] for entry in directional))
        directions = [int(score < threshold) for score in scores]
        mixed = archive.Spec("adaptive", 2, base, (row[0].grammars[0], column[0].grammars[0]))
        lengths = row[1][0], column[1][0]
        packets = [directional[direction][2][i] for i, direction in enumerate(directions)]
        costs = {codec: measure(mixed, directions, packets, codec, lengths if codec == "huffman" else None)
                 for codec in archive.CODECS}
        choices["adaptive"] = mixed, lengths, packets, costs
        spec = nd_spec(nd, budget)
        choices["nd"] = score_validation(spec, nd_packets(spec, validation))
        frozen[budget] = {"fixed_scan": selected, "threshold": threshold, "trials": trials, "models": {
            name: {"spec": asdict(entry[0]), "code_lengths": [table.hex() for table in entry[1]], "costs": entry[3]}
            for name, entry in choices.items()
        }}
    json_save(directory / "selected-before-test.json", frozen)
    return frozen


def unpack_model(entry):
    settings = entry["spec"]
    spec = archive.Spec(settings["mode"], settings["rank"], tuple(settings["base"]),
                        tuple(tuple(map(tuple, rules)) for rules in settings["grammars"]))
    return spec, tuple(bytes.fromhex(table) for table in entry["code_lengths"])


def test_packets(spec, samples, scores, threshold):
    if spec.mode != "adaptive":
        packets = nd_packets(spec, samples) if spec.mode == "nd" else linear_packets(spec, samples)
        return [0] * len(samples), packets
    directions = [int(score < threshold) for score in scores]
    ordered = [linear_packets(archive.Spec(mode, 2, spec.base, (rules,)), samples)
               for mode, rules in zip(("row", "column"), spec.grammars)]
    return directions, [ordered[direction][i] for i, direction in enumerate(directions)]


def measure_test(grids, directory):
    frozen = read_json(directory / "selected-before-test.json")
    samples = as_samples(grids)
    reference = reference_bytes(samples)
    scores = [neighbor_score(grid) for grid in grids]
    draws = np.random.default_rng(38004).integers(len(grids), size=(10_000, len(grids)))
    results = {}
    for budget, entry in frozen.items():
        files = {}
        fixed = entry["fixed_scan"]
        modes = sorted({*fixed.values(), "adaptive", "nd"})
        for mode in modes:
            spec, lengths = unpack_model(entry["models"][mode])
            directions, packets = test_packets(spec, samples, scores, entry["threshold"])
            files[mode] = {}
            for codec in archive.CODECS:
                if mode in ("adaptive", "nd") or fixed[codec] == mode:
                    files[mode][codec] = write_checked(directory / f"{mode}-{budget}-{codec}.mbpg", spec, directions,
                                                      packets, codec, lengths if codec == "huffman" else None, reference)
        comparisons = {}
        for codec, fixed_mode in fixed.items():
            baseline = files[fixed_mode][codec]
            comparisons[codec] = {method: differences(files[method][codec], baseline, draws)
                                  for method in ("adaptive", "nd")}
        results[budget] = {"fixed_scan": fixed, "threshold": entry["threshold"],
                           "row_choices": sum(score >= entry["threshold"] for score in scores),
                           "files": files, "comparisons": comparisons}
        print(f"{directory.name}/{budget}: {results[budget]['row_choices']} row choices; decoded all archives", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    process = constrain_cpu()
    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    data, manifest = freeze_inputs(args.backup)
    json_save(args.out / "inputs.json", manifest)
    for name, grids in data.items():
        directory = args.out / name
        directory.mkdir()
        prepare_models(grids, directory)
        print(f"{name}: validation choices frozen", flush=True)
    results = {name: measure_test(grids["test"], args.out / name) for name, grids in data.items()}
    memory = process.memory_info()
    sources = (Path(__file__), Path(archive.__file__), Path(learn_nd.__code__.co_filename))
    json_save(args.out / "report.json", {
        "complete": True, "results": results, "backup": str(args.backup.resolve()),
        "inputs_sha256": sha256((args.out / "inputs.json").read_bytes()),
        "source_sha256": {path.name: sha256(path.read_bytes()) for path in sources},
        "selection_sha256": {name: sha256((args.out / name / "selected-before-test.json").read_bytes()) for name in data},
        "seconds": time.perf_counter() - start, "python": sys.version, "numpy": np.__version__,
        "scipy": scipy.__version__, "zlib": zlib.ZLIB_RUNTIME_VERSION,
        "priority": int(process.nice()), "cpu_affinity": process.cpu_affinity(), "workers": 0,
        "peak_working_set": getattr(memory, "peak_wset", memory.rss),
    })


if __name__ == "__main__":
    main()
