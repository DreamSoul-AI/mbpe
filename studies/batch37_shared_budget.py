"""Keep the batch 37 policy fixed and match total grammar rules across methods."""

import argparse
from io import BytesIO
from pathlib import Path
import time

import psutil

import adaptive_bpe_file as adaptive
from batch30_mask_baselines import flatten
from batch33_celeba_files import json_save, sha256
from batch34_token_coding import apply_prefix, read_json
from batch35_shape_layouts import LAYOUTS, SIDE
from batch36_scan_orders import BUDGETS, CODECS
from batch37_adaptive_scan import (
    RESAMPLES, SEED, choose_direction, differences, load_inputs, neighbor_score, prepare_validation, write_checked,
)
from bpe_block_file import FileSpec, orders_for, write_archive, write_huffman_archive
from canonical_huffman import lengths_from_validation
import numpy as np


def prefix_tables(layouts, validation_tables):
    tables = {}
    for layout, grids in layouts.items():
        tables[layout] = {}
        for mode in adaptive.MODES:
            order = orders_for(SIDE)[mode]
            values = [flatten(grid, order) for grid in grids]
            for budget in BUDGETS:
                original, _ = validation_tables[layout][f"{mode}-{budget}"]
                spec = FileSpec(mode, original.side, original.alphabet, original.rules[:budget // 2])
                if original.rules != validation_tables[layout][f"{mode}-{max(BUDGETS)}"][0].rules[:budget]:
                    raise ValueError("expected nested directional grammars")
                lengths = lengths_from_validation(apply_prefix(spec, values), spec.symbols)
                tables[layout][mode, budget] = spec, lengths
    return tables


def measure_layout(directory, grids, manifest, layout, primary, threshold, tables, affinity):
    directory.mkdir()
    directions = [choose_direction(neighbor_score(grid), threshold) for grid in grids]
    if directions != primary["test_directions"]:
        raise ValueError("the total-budget control changed test directions")
    reference = b"".join(grid.tobytes() for grid in grids)
    order = orders_for(SIDE)
    originals = [[flatten(grid, order[mode]) for grid in grids] for mode in adaptive.MODES]
    draws = np.random.default_rng(SEED).integers(len(grids), size=(RESAMPLES, len(grids)))
    comparisons = {}
    for budget in BUDGETS:
        specs, lengths = zip(*(tables[mode, budget] for mode in adaptive.MODES))
        streams = [apply_prefix(spec, values) for spec, values in zip(specs, originals)]
        sequences = [streams[direction][i] for i, direction in enumerate(directions)]
        budget_rows = {}
        for codec in CODECS:
            fixed_costs = []
            for spec, stream, table in zip(specs, streams, lengths):
                buffer = BytesIO()
                cost = (write_huffman_archive(buffer, spec, stream, table) if codec == "huffman"
                        else write_archive(buffer, spec, stream))
                fixed_costs.append(cost)
            expected = [fixed_costs[direction]["payload_sizes"][i] for i, direction in enumerate(directions)]
            cost = write_checked(directory, f"adaptive-total-{budget}-{codec}", specs, directions, sequences,
                                 lengths if codec == "huffman" else None, expected, reference, manifest, layout, affinity)
            previous = primary["comparison"][str(budget)][codec]
            fixed = previous["baselines"][previous["fixed_scan"]]
            nd = previous["baselines"]["2d"]
            budget_rows[codec] = {"adaptive": cost, "fixed_scan": previous["fixed_scan"],
                                  "against_fixed": differences(cost, fixed, draws), "against_2d": differences(cost, nd, draws)}
            print(f"{layout}/{budget}/{codec}: {cost['tokens']} tokens, {cost['total_bytes']} bytes, "
                  f"{len(grids)} grids restored", flush=True)
        comparisons[str(budget)] = budget_rows
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source35", type=Path, required=True)
    parser.add_argument("--source36", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:1])
    if process.nice() != psutil.IDLE_PRIORITY_CLASS:
        raise RuntimeError("the experiment requires Idle priority")
    affinity = process.cpu_affinity()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    _, old36, data, manifest, fixed, provenance = load_inputs(args.source35, args.source36)
    primary_path = args.primary / "report.json"
    policy_path = args.primary / "selected-before-test.json"
    primary = read_json(primary_path)
    frozen = read_json(policy_path)
    policy, full_tables = prepare_validation(args.source36, data["validation"], old36)
    policy["fixed_scan"] = fixed
    if (not primary["complete"] or primary["provenance"] != provenance or frozen != primary["policy"]
            or frozen != policy):
        raise ValueError("expected the completed batch 37 report and its frozen selection")
    threshold = frozen["selection"]["threshold"]
    tables = prefix_tables(data["validation"], full_tables)
    config = {"primary_report_sha256": sha256(primary_path.read_bytes()), "policy_sha256": sha256(policy_path.read_bytes()),
              "source_sha256": sha256(Path(__file__).read_bytes()), "threshold": threshold, "total_rules": BUDGETS,
              "priority": int(process.nice()), "cpu_affinity": affinity, "data_workers": 0}
    json_save(args.out / "run-config.json", config)
    results = {}
    for layout in LAYOUTS:
        results[layout] = measure_layout(args.out / layout, data["test"][layout], manifest, layout,
                                         primary["results"][layout], threshold, tables[layout], affinity)
    memory = process.memory_info()
    json_save(args.out / "report.json", {**config, "complete": True, "results": results,
              "seconds": time.perf_counter() - started, "peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss)})


if __name__ == "__main__":
    main()
