"""Freeze a validation-selected row/column policy and measure complete BPE files."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import zlib

import psutil

import adaptive_bpe_file as adaptive
from batch33_celeba_files import STUDIES, json_save, sha256
from batch34_token_coding import check_decoded_grids, read_json
from batch35_shape_layouts import LAYOUTS, SIDE, freeze_pairs
from batch36_scan_orders import BUDGETS, CODECS, load_previous
from bpe_block_file import ORDERS
import numpy as np


SEED = 37004
RESAMPLES = 10_000
BATCH35_REPORT = "ac5009144fc29675e7b68954e1430f51a0a5064a935efd6759a24779bc5db874"
BATCH36_REPORT = "0749edcd145eb4271e7e1b285f15539e46ed81fd35f14e032ee5eaf4d9a15114"


def neighbor_score(grid):
    horizontal = np.count_nonzero(grid[:, :-1] == grid[:, 1:])
    vertical = np.count_nonzero(grid[:-1, :] == grid[1:, :])
    return int(horizontal - vertical)


def choose_direction(score, threshold):
    return int(score < threshold)  # 0 is row, 1 is column.


def fit_threshold(scores, validation_costs):
    pairs = []
    for layout, values in scores.items():
        costs = validation_costs[layout]
        for budget in BUDGETS:
            sizes = [costs[f"{mode}-{budget}"]["huffman"]["payload_sizes"] for mode in adaptive.MODES]
            pairs.extend(zip(values, *sizes, strict=True))
    seen = {score for score, _, _ in pairs}
    thresholds = sorted(seen | {0, max(seen) + 1})
    trials = [
        {"threshold": threshold, "huffman_payload_bytes": sum(
            row if score >= threshold else column for score, row, column in pairs
        )}
        for threshold in thresholds
    ]
    best = min(trials, key=lambda trial: (
        trial["huffman_payload_bytes"], abs(trial["threshold"]), trial["threshold"],
    ))
    return best, trials


def load_inputs(source35, source36):
    old35, old36 = (read_json(source / "report.json") for source in (source35, source36))
    reports = [sha256((source / "report.json").read_bytes()) for source in (source35, source36)]
    if reports != [BATCH35_REPORT, BATCH36_REPORT] or not (old35["complete"] and old36["complete"]):
        raise ValueError("expected the completed, frozen batch 35 and 36 reports")
    manifest_path = source35 / "inputs.json"
    manifest = read_json(manifest_path)
    data, regenerated = freeze_pairs()
    if (sha256(manifest_path.read_bytes()) != old35["inputs_sha256"]
            or old35["inputs_sha256"] != old36["inputs_sha256"]
            or old36["batch35_report_sha256"] != BATCH35_REPORT
            or manifest != json.loads(json.dumps(regenerated))
            or not old35["zlib"] == old36["zlib"] == zlib.ZLIB_RUNTIME_VERSION):
        raise ValueError("frozen input or zlib runtime differs from the prior runs")
    dependencies = ("batch37_adaptive_scan.py", "adaptive_bpe_file.py", *old36["source_sha256"])
    hashes = {name: sha256((STUDIES / name).read_bytes()) for name in dependencies}
    if any(hashes[name] != digest for name, digest in old36["source_sha256"].items()):
        raise ValueError("frozen source dependency changed")
    selected_path = source36 / "selected-before-test.json"
    fixed = read_json(selected_path)
    if any(fixed[layout] != old36["results"][layout]["selected"] for layout in LAYOUTS):
        raise ValueError("prior validation selections do not match the batch 36 report")
    provenance = {"batch35_report_sha256": reports[0], "batch36_report_sha256": reports[1],
                  "batch35_inputs_sha256": old35["inputs_sha256"], "source_sha256": hashes,
                  "batch36_selection_sha256": sha256(selected_path.read_bytes())}
    return old35, old36, data, manifest, fixed, provenance


def prepare_validation(source, layouts, report):
    scores, validation_costs, tables = {}, {}, {}
    for layout, grids in layouts.items():
        scores[layout] = [neighbor_score(grid) for grid in grids]
        reference = b"".join(grid.tobytes() for grid in grids)
        costs = report["results"][layout]["validation"]
        validation_costs[layout] = costs
        tables[layout] = {}
        for mode in adaptive.MODES:
            for budget in BUDGETS:
                key = f"{mode}-{budget}"
                spec, lengths, _ = load_previous(source / layout / "validation", key, costs[key],
                                                  reference, budget, mode)
                tables[layout][key] = spec, lengths
    best, trials = fit_threshold(scores, validation_costs)
    policy = {"selection": best, "trials": trials, "validation_scores": scores}
    return policy, tables


def write_checked(directory, name, specs, directions, sequences, lengths, expected_sizes,
                  reference, manifest, layout, affinity):
    archive = directory / f"{name}.mbpa"
    decoded = directory / f"{name}.decoded"
    with archive.open("xb") as handle:
        cost = adaptive.write_archive(handle, specs, directions, sequences, lengths)
    with archive.open("rb") as handle:
        parsed = adaptive.read_token_archive(handle)
    if parsed != (specs, lengths, directions, sequences, cost):
        raise ValueError(f"serialized grammar, directions, tokens, or accounting differ: {archive}")
    if cost["payload_sizes"] != expected_sizes or cost["total_bytes"] != archive.stat().st_size:
        raise ValueError(f"unexpected block or file size: {archive}")

    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", str(STUDIES / "adaptive_bpe_file.py"), str(archive), str(decoded)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    receipt = directory / f"{name}.json"
    if proc.returncode:
        json_save(receipt, {"decoder_exit_code": proc.returncode, "decoder_error": proc.stderr})
        raise ValueError(f"independent decoder failed: {archive}")
    result = json.loads(proc.stdout)
    if any(result[key] != value for key, value in cost.items()):
        raise ValueError(f"independent decoder accounting differs: {archive}")
    if result["priority"] != int(psutil.IDLE_PRIORITY_CLASS) or result["cpu_affinity"] != affinity:
        raise ValueError("independent decoder is outside the experiment's CPU limits")
    recovered = decoded.read_bytes()
    cost.update(
        file=archive.name, file_sha256=sha256(archive.read_bytes()), decoded_sha256=sha256(recovered),
        tokens=sum(cost["token_counts"]), lossless_bad=check_decoded_grids(recovered, manifest, layout, "test", SIDE),
        decoder_priority=result["priority"], decoder_cpu_affinity=result["cpu_affinity"],
        decoder_peak_working_set_bytes=result["peak_working_set_bytes"], decoder_seconds=time.perf_counter() - started,
    )
    json_save(receipt, cost)
    if recovered != reference or cost["lossless_bad"]:
        raise ValueError(f"independent decoder differs from frozen grids: {archive}")
    return cost


def differences(mixed, baseline, draws):
    metrics = {}
    for metric, field in (("tokens", "token_counts"), ("bytes", "payload_sizes")):
        per_image = np.asarray(mixed[field], dtype=np.int64) - baseline[field]
        total = int(per_image.sum())
        fixed = mixed["total_bytes"] - baseline["total_bytes"] - total if metric == "bytes" else 0
        metrics[metric] = {"total": fixed + total, "fixed": fixed, "per_image": per_image.tolist(),
                           "bootstrap_95": np.quantile(fixed + per_image[draws].sum(axis=1), (0.025, 0.975)).tolist()}
    return metrics


def measure_layout(args, layout, grids, manifest, policy, tables, costs35, costs36, fixed, affinity):
    scores = [neighbor_score(grid) for grid in grids]
    directions = [choose_direction(score, policy["selection"]["threshold"]) for score in scores]
    directory = args.out / layout
    directory.mkdir()
    json_save(directory / "test-directions.json", {"scores": scores, "directions": directions})

    reference = b"".join(grid.tobytes() for grid in grids)
    draws = np.random.default_rng(SEED).integers(len(grids), size=(RESAMPLES, len(grids)))
    results = {}
    for budget in BUDGETS:
        rows, tables_and_tokens = {}, {}
        for mode in (*ORDERS, "2d"):
            key = f"{mode}-{budget}"
            if mode in ("hilbert", "2d"):
                source = args.source35 / layout
                costs = costs35[key]
            else:
                source = args.source36 / layout / "test"
                costs = costs36[key]
            spec, lengths, tokens = load_previous(source, key, costs, reference, budget, mode)
            if mode in adaptive.MODES and (spec, lengths) != tables[key]:
                raise ValueError(f"test grammar or code table differs from validation: {layout}/{key}")
            rows[mode] = costs
            tables_and_tokens[mode] = spec, lengths, tokens

        specs, lengths, streams = zip(*(tables_and_tokens[mode] for mode in adaptive.MODES))
        sequences = [streams[direction][i] for i, direction in enumerate(directions)]
        budget_rows = {}
        for codec in CODECS:
            sizes = [rows[adaptive.MODES[direction]][codec]["payload_sizes"][i]
                     for i, direction in enumerate(directions)]
            name = f"adaptive-{budget}-{codec}"
            cost = write_checked(directory, name, specs, directions, sequences,
                                 lengths if codec == "huffman" else None, sizes, reference, manifest, layout, affinity)
            selected = fixed["all"][str(budget)][codec]
            baselines = {mode: values[codec] for mode, values in rows.items()}
            budget_rows[codec] = {"adaptive": cost, "fixed_scan": selected, "baselines": baselines,
                                  "against_fixed": differences(cost, baselines[selected], draws),
                                  "against_2d": differences(cost, baselines["2d"], draws)}
            print(f"{layout}/{budget}/{codec}: {cost['tokens']} tokens, {cost['total_bytes']} bytes, "
                  f"{len(grids)} grids restored", flush=True)
        results[str(budget)] = budget_rows
    return {"test_scores": scores, "test_directions": directions, "comparison": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source35", type=Path, required=True)
    parser.add_argument("--source36", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:1])
    if process.nice() != psutil.IDLE_PRIORITY_CLASS:
        raise RuntimeError("the experiment requires Idle priority")
    affinity = process.cpu_affinity()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    old35, old36, data, manifest, fixed, provenance = load_inputs(args.source35, args.source36)
    report = {"complete": False, "provenance": provenance,
              "source35": str(args.source35.resolve()), "source36": str(args.source36.resolve()),
              "python": sys.version, "numpy": np.__version__, "zlib": zlib.ZLIB_RUNTIME_VERSION,
              "psutil": psutil.__version__, "priority": int(process.nice()), "cpu_affinity": affinity,
              "data_workers": 0, "budgets": BUDGETS, "bootstrap_seed": SEED, "bootstrap_repetitions": RESAMPLES}
    json_save(args.out / "run-config.json", report)
    policy, tables = prepare_validation(args.source36, data["validation"], old36)
    policy["fixed_scan"] = fixed
    json_save(args.out / "selected-before-test.json", policy)
    print(f"Validation threshold frozen: {policy['selection']}", flush=True)

    results = {}
    for layout in LAYOUTS:
        results[layout] = measure_layout(
            args, layout, data["test"][layout], manifest, policy, tables[layout], old35["results"][layout]["files"],
            old36["results"][layout]["test"], fixed[layout], affinity,
        )
        json_save(args.out / layout / "summary.json", results[layout])
    memory = process.memory_info()
    report.update(complete=True, policy=policy, results=results, seconds=time.perf_counter() - started,
                  peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss))
    json_save(args.out / "report.json", report)
    print(f"Complete in {report['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
