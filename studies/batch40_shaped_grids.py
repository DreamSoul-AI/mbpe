"""Compare complete linear and geometric BPE files on non-square and sparse grids."""

import argparse
from dataclasses import asdict
from pathlib import Path
import pickle
import sys
import time
import zlib

from batch33_celeba_files import json_save, sha256
from batch34_token_coding import read_json
from batch37_adaptive_scan import differences
from batch38_native_grids import BUDGETS, score_validation, unpack_model
import grid_bpe_file as archive
from grid_bpe_study import (
    constrain_cpu, learn_linear, learn_nd, linear_packets, nd_packets, nd_spec, reference_bytes, write_checked,
)
import numpy as np


SCANS = ("row", "column", "snake")
LAYOUTS = ("rectangle", "holes", "rgb")
PER_CLASS = {"learn": 4, "validation": 2, "test": 4}


def samples_from_source(batch38, backup):
    report = read_json(batch38 / "report.json")
    raw = (batch38 / "inputs.json").read_bytes()
    if not report["complete"] or sha256(raw) != report["inputs_sha256"]:
        raise ValueError("batch 38 input manifest changed")
    previous = read_json(batch38 / "inputs.json")["cifar"]
    cached, layouts, manifest = {}, {name: {} for name in LAYOUTS}, {}
    for split, source in previous.items():
        path = backup / source["source"]
        if path not in cached:
            blob = path.read_bytes()
            if sha256(blob) != source["source_sha256"]:
                raise ValueError(f"CIFAR source changed: {path}")
            row = pickle.loads(blob, encoding="bytes")
            cached[path] = row[b"data"].reshape(-1, 3, 32, 32)
        rows = [row for label in range(10) for row in
                [item for item in source["samples"] if item["class"] == label][:PER_CLASS[split]]]
        if len(rows) != 10 * PER_CLASS[split]:
            raise ValueError("not enough frozen input grids")
        manifest[split] = {"source": source["source"], "source_sha256": source["source_sha256"], "samples": []}
        for name in LAYOUTS:
            layouts[name][split] = []
        for row in rows:
            channels = cached[path][row["index"]]
            if sha256(channels[1].tobytes()) != row["sha256"]:
                raise ValueError("CIFAR grid differs from the frozen input")
            crops = {"rectangle": channels[1, 4:28, :], "holes": channels[1, 4:28, :],
                     "rgb": channels[:, 8:24, 4:28].transpose(1, 2, 0)}
            hashes = {}
            for name, crop in crops.items():
                cells = {point: int(value) for point, value in np.ndenumerate(crop)}
                if name == "holes":
                    for y, x in ((3 + row["index"] % 3, 4), (13, 19 + row["index"] % 3)):
                        for dy, dx in archive.grid_points((6, 8)):
                            del cells[y + dy, x + dx]
                sample = crop.shape, cells
                layouts[name][split].append(sample)
                hashes[name] = sha256(reference_bytes([sample]))
            manifest[split]["samples"].append({**row, "shaped_sha256": hashes})
    return layouts, manifest


def prepare(directory, splits):
    learn, validation = splits["learn"], splits["validation"]
    rank = len(learn[0][0])
    base = tuple(sorted({value for _, cells in learn for value in cells.values()}))
    linear = {mode: learn_linear(learn, base, mode, max(BUDGETS)) for mode in SCANS}
    nd = learn_nd(learn, max(BUDGETS))
    choices = {}
    for budget in BUDGETS:
        candidates = {}
        for mode, rules in linear.items():
            spec = archive.Spec(mode, rank, base, (rules[:budget],))
            candidates[mode] = score_validation(spec, linear_packets(spec, validation))
        selected = {codec: min(SCANS, key=lambda mode: (
            candidates[mode][3][codec]["total_bytes"], candidates[mode][3][codec]["tokens"], mode,
        )) for codec in archive.CODECS}
        spec = nd_spec(nd, budget)
        candidates["nd"] = score_validation(spec, nd_packets(spec, validation))
        choices[budget] = {"fixed_scan": selected, "models": {
            name: {"spec": asdict(entry[0]), "code_lengths": [table.hex() for table in entry[1]], "costs": entry[3]}
            for name, entry in candidates.items()
        }}
    json_save(directory / "selected-before-test.json", choices)


def measure_test(directory, samples):
    choices = read_json(directory / "selected-before-test.json")
    directions = [0] * len(samples)
    reference = reference_bytes(samples)
    draws = np.random.default_rng(40004).integers(len(samples), size=(10_000, len(samples)))
    output = {}
    for budget, row in choices.items():
        fixed = row["fixed_scan"]
        costs = {}
        for mode in sorted({*fixed.values(), "nd"}):
            spec, lengths = unpack_model(row["models"][mode])
            packets = nd_packets(spec, samples) if mode == "nd" else linear_packets(spec, samples)
            costs[mode] = {}
            for codec in archive.CODECS:
                if mode != "nd" and mode != fixed[codec]:
                    continue
                name = f"{mode}-{budget}-{codec}"
                costs[mode][codec] = write_checked(directory / f"{name}.mbpg", spec, directions, packets, codec,
                                                   lengths if codec == "huffman" else None, reference)
                if codec == "zlib" and budget == "0":
                    width = max(9, max(costs[mode][codec]["widths"]))
                    costs[mode]["zlib-common"] = write_checked(directory / f"{name}-width{width}.mbpg", spec, directions,
                                                               packets, codec, None, reference, width)
        output[budget] = {"fixed_scan": fixed, "files": costs, "comparisons": {
            codec: differences(costs["nd"][codec], costs[mode][codec], draws) for codec, mode in fixed.items()
        }}
    return {"cells": sum(len(cells) for _, cells in samples), "samples": len(samples), "results": output}


def edge_samples(rank):
    if rank == 0:
        return [((), {(): -(1 << 85)}), ((), {})]
    return [((0, 5), {}), ((3, 4), {}), ((2, 3), {(0, 0): -17, (0, 2): 4, (1, 1): 1 << 90})]


def measure_edges(directory):
    directory.mkdir()
    rows = {}
    for rank in (0, 2):
        samples = edge_samples(rank)
        reference = reference_bytes(samples)
        directions = [0] * len(samples)
        for mode in ("row", "nd"):
            spec = archive.Spec(mode, rank, (4,), ((),))
            packets = nd_packets(spec, samples) if mode == "nd" else linear_packets(spec, samples)
            validation = nd_packets(spec, [((), {(): 4})]) if rank == 0 and mode == "nd" else []
            tables = archive.code_lengths(spec, [0] * len(validation), validation)
            for codec in archive.CODECS:
                name = f"rank{rank}-{mode}-{codec}"
                rows[name] = write_checked(directory / f"{name}.mbpg", spec, directions, packets, codec,
                                           tables if codec == "huffman" else None, reference)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("batch38", "backup", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    process = constrain_cpu()
    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    layouts, manifest = samples_from_source(args.batch38, args.backup)
    json_save(args.out / "inputs.json", manifest)
    results = {}
    for name, splits in layouts.items():
        target = args.out / name
        target.mkdir()
        prepare(target, splits)
        results[name] = measure_test(target, splits["test"])
        print(f"{name}: {results[name]['samples']} grids; decoded all archives", flush=True)
    edges = measure_edges(args.out / "edges")
    memory = process.memory_info()
    json_save(args.out / "report.json", {
        "complete": True, "layouts": results, "edge_cases": edges, "seconds": time.perf_counter() - start,
        "inputs_sha256": sha256((args.out / "inputs.json").read_bytes()),
        "batch38_inputs_sha256": sha256((args.batch38 / "inputs.json").read_bytes()),
        "source_sha256": {path.name: sha256(path.read_bytes()) for path in
                          (Path(__file__), Path(archive.__file__), Path(write_checked.__code__.co_filename),
                           Path(score_validation.__code__.co_filename))},
        "python": sys.version, "numpy": np.__version__, "zlib": zlib.ZLIB_RUNTIME_VERSION,
        "priority": int(process.nice()), "cpu_affinity": process.cpu_affinity(), "workers": 0,
        "peak_working_set": getattr(memory, "peak_wset", memory.rss),
    })


if __name__ == "__main__":
    main()
