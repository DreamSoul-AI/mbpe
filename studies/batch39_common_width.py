"""Recode frozen BPE streams using one common token width per comparison."""

import argparse
from pathlib import Path
import time
import zlib

import adaptive_bpe_file as old_adaptive
import bpe_block_file as old_fixed
from batch33_celeba_files import json_save, sha256
from batch34_token_coding import read_json
from batch37_adaptive_scan import differences
import grid_bpe_file as archive
from grid_bpe_study import constrain_cpu, write_checked
import numpy as np


def verify(path, expected):
    blob = path.read_bytes()
    if sha256(blob) != expected:
        raise ValueError(f"source file changed: {path}")
    return blob


def frozen_file(path, cost):
    verify(path, cost["sha256"])
    reference = verify(path.with_suffix(".decoded"), cost["decoded_sha256"])
    with path.open("rb") as handle:
        spec, directions, packets, codec, _, width, actual = archive.read_archive(handle)
    if codec != "zlib" or width or any(cost[key] != value for key, value in actual.items()):
        raise ValueError(f"native source accounting differs: {path}")
    return spec, directions, packets, reference


def width_for(stream):
    spec, directions, packets, _ = stream
    return max(archive.bit_width(spec.symbols(direction) + len(packet.literals))
               for direction, packet in zip(directions, packets, strict=True))


def recode(directory, name, original, stream, width):
    spec, directions, packets, reference = stream
    result = write_checked(directory / f"{name}.mbpg", spec, directions, packets, "zlib", None, reference, width)
    if result["token_counts"] != original["token_counts"]:
        raise ValueError(f"repacking changed the number of tokens: {name}")
    if "widths" in original:
        fixed = lambda row: row["total_bytes"] - row["payload_bytes"]
        if fixed(result) != fixed(original):
            raise ValueError(f"repacking changed non-payload bytes: {name}")
    return result


def summarize(files):
    baseline = files["fixed"]
    count = len(baseline["token_counts"])
    draws = np.random.default_rng(39004).integers(count, size=(10_000, count))
    return {name: differences(cost, baseline, draws) for name, cost in files.items() if name != "fixed"}


def native(source, directory):
    report = read_json(source / "report.json")
    if not report["complete"] or report["zlib"] != zlib.ZLIB_RUNTIME_VERSION:
        raise ValueError("expected a complete batch 38 report with the same zlib version")
    output = {}
    for dataset, budgets in report["results"].items():
        target = directory / dataset
        target.mkdir()
        streams, measured = {}, {}
        for budget, row in budgets.items():
            modes = {"fixed": row["fixed_scan"]["zlib"], "adaptive": "adaptive", "nd": "nd"}
            measured[budget] = {name: row["files"][mode]["zlib"] for name, mode in modes.items()}
            streams[budget] = {name: frozen_file(source / dataset / cost["file"], cost)
                               for name, cost in measured[budget].items()}
        all_width = max(width_for(stream) for methods in streams.values() for stream in methods.values())
        output[dataset] = {"all_width": all_width, "budgets": {}}
        for budget, methods in streams.items():
            same_budget_width = max(map(width_for, methods.values()))
            normalized = {}
            for width in sorted({same_budget_width, all_width}):
                normalized[width] = {}
                for name, stream in methods.items():
                    original = measured[budget][name]
                    if set(original["widths"]) == {width}:
                        normalized[width][name] = {**original, "reused": True}
                    else:
                        normalized[width][name] = recode(target, f"{name}-{budget}-width{width}", original, stream, width)
            output[dataset]["budgets"][budget] = {
                "original": measured[budget], "by_width": normalized, "same_budget_width": same_budget_width,
                "comparisons": {width: summarize(rows) for width, rows in normalized.items()},
            }
        print(f"{dataset}: normalized budgets at up to {all_width} bits", flush=True)
    return output


def old_stream(path, cost, adaptive):
    verify(path, cost["file_sha256"])
    raw = verify(path.with_suffix(".decoded"), cost["decoded_sha256"])
    with path.open("rb") as handle:
        if adaptive:
            specs, lengths, directions, sequences, actual = old_adaptive.read_token_archive(handle)
            mode, spec = "adaptive", specs[0]
        else:
            spec, sequences, actual = old_fixed.read_token_archive(handle)
            specs, lengths, directions = (spec,), None, [0] * len(sequences)
            mode = "nd" if spec.mode == "2d" else spec.mode
    if lengths is not None or any(cost[key] != value for key, value in actual.items()):
        raise ValueError(f"synthetic source accounting differs: {path}")
    shape = (spec.side, spec.side)
    general = archive.Spec(mode, 2, tuple(range(spec.alphabet)), tuple(item.rules for item in specs))
    packets = [archive.Packet(shape, None, (), tuple(values)) for values in sequences]
    reference = b"".join(archive.grid_bytes(shape, None, list(raw[i:i + spec.cells]))
                         for i in range(0, len(raw), spec.cells))
    return general, directions, packets, reference


def synthetic(primary, shared, directory):
    main, matched = (read_json(source / "report.json") for source in (primary, shared))
    if not (main["complete"] and matched["complete"]) or main["zlib"] != zlib.ZLIB_RUNTIME_VERSION:
        raise ValueError("expected completed batch 37 sources with the same zlib version")
    verify(primary / "report.json", matched["primary_report_sha256"])
    source35, source36 = (Path(main[key]) for key in ("source35", "source36"))
    output = {}
    for layout, entry in main["results"].items():
        target = directory / layout
        target.mkdir()
        output[layout] = {}
        for budget, rows in entry["comparison"].items():
            row = rows["zlib"]
            fixed = row["fixed_scan"]
            fixed_dir = source35 / layout if fixed == "hilbert" else source36 / layout / "test"
            sources = {"fixed": (fixed_dir, row["baselines"][fixed]),
                       "nd": (source35 / layout, row["baselines"]["2d"]),
                       "adaptive": (shared / layout, matched["results"][layout][budget]["zlib"]["adaptive"])}
            streams = {name: old_stream(path / cost["file"], cost, name == "adaptive")
                       for name, (path, cost) in sources.items()}
            if len({stream[3] for stream in streams.values()}) != 1:
                raise ValueError(f"synthetic methods do not share the same grids: {layout}/{budget}")
            common = max(map(width_for, streams.values()))
            original, normalized = {}, {}
            for name, stream in streams.items():
                old = sources[name][1]
                natural = recode(target, f"{name}-{budget}-native", old, stream, 0)
                if natural["payload_sizes"] != old["payload_sizes"]:
                    raise ValueError(f"migration changed native payload lengths: {layout}/{budget}/{name}")
                original[name] = natural
                normalized[name] = (natural if width_for(stream) == common else
                                    recode(target, f"{name}-{budget}-width{common}", natural, stream, common))
            output[layout][budget] = {"original_mbpe": {name: cost for name, (_, cost) in sources.items()},
                                      "original": original, "common_width": common, "normalized": normalized,
                                      "against_fixed": summarize(normalized)}
        print(f"{layout}: checked frozen synthetic files", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "primary37", "shared37", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    process = constrain_cpu()
    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    observed = native(args.native, args.out)
    controlled = synthetic(args.primary37, args.shared37, args.out)
    json_save(args.out / "report.json", {
        "complete": True, "native": observed, "synthetic": controlled, "seconds": time.perf_counter() - start,
        "source_reports": {str(root): sha256((root / "report.json").read_bytes())
                           for root in (args.native, args.primary37, args.shared37)},
        "source_sha256": {path.name: sha256(path.read_bytes()) for path in
                          (Path(__file__), Path(archive.__file__), Path(write_checked.__code__.co_filename))},
        "zlib": zlib.ZLIB_RUNTIME_VERSION, "priority": int(process.nice()), "cpu_affinity": process.cpu_affinity(), "workers": 0,
    })


if __name__ == "__main__":
    main()
