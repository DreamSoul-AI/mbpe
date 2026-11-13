"""Compare equal-budget BPE using the frozen batch 35 layouts."""

import argparse
import json
from pathlib import Path
import sys
import time
import zlib

from batch30_mask_baselines import flatten, learn_apply, pick
from batch33_celeba_files import STUDIES, json_save, sha256
from batch34_token_coding import apply_prefix, read_json, write_checked
from batch35_shape_layouts import ALPHABET, LAYOUTS, SIDE, freeze_pairs
from bpe_block_file import HUFFMAN_VERSION, ORDERS, VERSION, FileSpec, orders_for, read_exact, read_prefix
from bpe_block_file import read_token_archive, write_archive, write_huffman_archive
from canonical_huffman import lengths_from_validation
import numpy as np
import psutil


BUDGETS = (64, 256)
CODECS = ("huffman", "zlib")
SEED = 36004
RESAMPLES = 10_000


def subsets(manifest, layout, split):
    rows = manifest["splits"][split]
    groups = {"all": list(range(len(rows)))}
    if layout == "stripe":
        groups.update(
            horizontal=[i for i, row in enumerate(rows) if not row["transposed"]],
            vertical=[i for i, row in enumerate(rows) if row["transposed"]],
        )
    return groups


def write_files(directory, key, spec, lengths, sequences, reference, manifest, layout, split, codecs=CODECS):
    writers = {
        "huffman": lambda handle: write_huffman_archive(handle, spec, sequences, lengths),
        "zlib": lambda handle: write_archive(handle, spec, sequences),
    }
    return {
        codec: write_checked(directory, f"{key}-{codec}", writers[codec], spec,
                             sequences, reference, manifest, layout, split)
        for codec in codecs
    }


def load_previous(source, key, costs, reference, budget, mode):
    sequences, lengths, saved_spec = None, None, None
    for codec in CODECS:
        expected = costs[codec]
        archive = source / expected["file"]
        if sha256(archive.read_bytes()) != expected["file_sha256"]:
            raise ValueError(f"changed batch 35 file: {archive}")
        with archive.open("rb") as handle:
            spec, _, version, _, _ = read_prefix(handle)
            if codec == "huffman" and version == HUFFMAN_VERSION:
                lengths = read_exact(handle, spec.symbols)
            elif codec != "zlib" or version != VERSION:
                raise ValueError(f"unexpected file version: {archive}")
            handle.seek(0)
            read_spec, tokens, cost = read_token_archive(handle)
        if spec != read_spec or (spec.mode, spec.side, spec.alphabet, len(spec.rules)) != (mode, SIDE, ALPHABET, budget):
            raise ValueError(f"changed batch 35 grammar: {key}")
        if saved_spec is not None and (saved_spec != spec or sequences != tokens):
            raise ValueError(f"codecs disagree on the original token stream: {key}")
        if cost["total_bytes"] != archive.stat().st_size or any(expected[k] != v for k, v in cost.items()):
            raise ValueError(f"changed batch 35 file accounting: {archive}")
        if expected["lossless_bad"] or archive.with_suffix(".decoded").read_bytes() != reference:
            raise ValueError(f"changed batch 35 decoded grids: {archive}")
        saved_spec, sequences = spec, tokens
    return saved_spec, lengths, sequences


def group_cost(row, indices):
    # The shared section is unchanged and every image has an independent block.
    payloads = [row["payload_sizes"][i] for i in indices]
    counts = [row["token_counts"][i] for i in indices]
    per_image = row["block_header_bytes"] // row["images"]
    fixed = row["total_bytes"] - row["payload_bytes"] - row["block_header_bytes"]
    return {"tokens": sum(counts), "token_counts": counts,
            "payload_sizes": payloads, "total_bytes": fixed + per_image * len(indices) + sum(payloads)}


def prepare_layout(directory, data, manifest, layout, old):
    validation_dir = directory / "validation"
    validation_dir.mkdir(parents=True)
    learn, validation = (data[split][layout] for split in ("learn", "validation"))
    reference = b"".join(image.tobytes() for image in validation)
    specs, costs = {}, {}
    for mode in ORDERS:
        order = orders_for(SIDE)[mode]
        flat_validation = [flatten(image, order) for image in validation]
        if mode == "hilbert":
            rules = read_json(old / "hilbert-rules.json")
        else:
            rules, checkpoints = learn_apply(
                [flatten(image, order) for image in learn], {"validation": flat_validation},
                ALPHABET, max(BUDGETS), set(BUDGETS),
            )
            json_save(directory / f"{mode}-rules.json", rules)
        if len(rules) != max(BUDGETS):
            raise ValueError(f"could not reach the fixed budget: {layout}/{mode}")
        for budget in BUDGETS:
            key = f"{mode}-{budget}"
            spec = FileSpec(mode, SIDE, ALPHABET, tuple((left, right) for left, right, _ in rules[:budget]))
            if mode == "hilbert":
                sequences = apply_prefix(spec, flat_validation)
            else:
                (sequences, failures), used = pick(checkpoints["validation"], budget)
                if failures or used != budget:
                    raise ValueError(f"failed validation or wrong budget: {layout}/{key}")
            lengths = lengths_from_validation(sequences, spec.symbols)
            specs[key] = (spec, lengths)
            costs[key] = write_files(validation_dir, key, spec, lengths, sequences, reference, manifest,
                                     layout, "validation")
    selected, selection_costs = {}, {}
    for group, indices in subsets(manifest, layout, "validation").items():
        selected[group], selection_costs[group] = {}, {}
        for budget in BUDGETS:
            selections, scores = {}, {}
            for codec in CODECS:
                ranking = {
                    mode: group_cost(costs[f"{mode}-{budget}"][codec], indices) for mode in ORDERS
                }
                selections[codec] = min(ranking, key=lambda mode: (
                    ranking[mode]["total_bytes"], ranking[mode]["tokens"], mode,
                ))
                scores[codec] = {mode: {key: row[key] for key in ("total_bytes", "tokens")}
                                 for mode, row in ranking.items()}
            selected[group][str(budget)] = selections
            selection_costs[group][str(budget)] = scores
    json_save(directory / "selection.json", {"selected": selected, "validation_scores": selection_costs})
    return specs, selected, selection_costs, costs


def differences(nd, linear):
    metrics = {}
    for metric, field in (("tokens", "token_counts"), ("bytes", "payload_sizes")):
        per_image = np.asarray(nd[field], dtype=np.int64) - linear[field]
        fixed = 0
        if metric == "bytes":
            fixed = nd["total_bytes"] - linear["total_bytes"] - int(per_image.sum())
        draws = np.random.default_rng(SEED).integers(len(per_image), size=(RESAMPLES, len(per_image)))
        metrics[metric] = {
            "total": fixed + int(per_image.sum()), "fixed": fixed, "per_image": per_image.tolist(),
            "bootstrap_95": np.quantile(fixed + per_image[draws].sum(axis=1), (0.025, 0.975)).tolist(),
        }
    return metrics


def measure_layout(directory, data, manifest, layout, old, old_costs, prepared):
    specs, selected, selection_costs, validation_costs = prepared
    images = data["test"][layout]
    reference = b"".join(image.tobytes() for image in images)
    test_dir = directory / "test"
    test_dir.mkdir()
    test_costs, streams = {}, {}
    for budget in BUDGETS:
        for mode in (*ORDERS, "2d"):
            key = f"{mode}-{budget}"
            if mode in ("hilbert", "2d"):
                costs = old_costs[key]
                spec, lengths, sequences = load_previous(old, key, costs, reference, budget, mode)
                if mode == "hilbert" and (spec, lengths) != specs[key]:
                    raise ValueError(f"reconstructed Hilbert validation differs from batch 35: {key}")
            else:
                spec, lengths = specs[key]
                order = orders_for(SIDE)[mode]
                sequences = apply_prefix(spec, [flatten(image, order) for image in images])
                costs = write_files(test_dir, key, spec, lengths, sequences, reference, manifest, layout, "test")
            test_costs[key] = costs
            streams[key] = (spec, lengths, sequences)

    comparison = {}
    for group, indices in subsets(manifest, layout, "test").items():
        group_dir = test_dir / group
        if group != "all":
            group_dir.mkdir()
        group_reference = b"".join(images[i].tobytes() for i in indices)
        group_manifest = {"splits": {"test": [manifest["splits"]["test"][i] for i in indices]}}
        comparison[group] = {}
        for budget in BUDGETS:
            comparison[group][str(budget)] = {}
            nd_key = f"2d-{budget}"
            for codec in CODECS:
                selected_mode = selected[group][str(budget)][codec]
                linear_key = f"{selected_mode}-{budget}"
                rows = {method: group_cost(test_costs[f"{method}-{budget}"][codec], indices)
                        for method in (*ORDERS, "2d")}
                subgroup_files = {}
                if group != "all":
                    for key in (nd_key, linear_key):
                        spec, lengths, sequences = streams[key]
                        cost = write_files(group_dir, key, spec, lengths, [sequences[i] for i in indices],
                                           group_reference, group_manifest, layout, "test", (codec,))[codec]
                        if group_cost(cost, range(len(indices))) != rows[spec.mode]:
                            raise ValueError(f"subgroup bytes differ from the measured file: {key}/{codec}")
                        subgroup_files[key] = cost
                comparison[group][str(budget)][codec] = {
                    "selected": selected_mode, "rows": rows, "subgroup_files": subgroup_files,
                    "against_selected": differences(rows["2d"], rows[selected_mode]),
                    "against_hilbert": differences(rows["2d"], rows["hilbert"]),
                }
    return {"validation": validation_costs, "test": test_costs, "selected": selected,
            "validation_scores": selection_costs, "comparison": comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:1])
    started = time.perf_counter()
    old = read_json(args.source / "report.json")
    manifest_path = args.source / "inputs.json"
    manifest = read_json(manifest_path)
    data, regenerated = freeze_pairs()
    if (not old["complete"] or sha256(manifest_path.read_bytes()) != old["inputs_sha256"]
            or manifest != json.loads(json.dumps(regenerated)) or old["zlib"] != zlib.ZLIB_RUNTIME_VERSION):
        raise ValueError("inputs or environment differ from batch 35")
    dependencies = ("batch36_scan_orders.py", *old["source_sha256"])
    source_hashes = {name: sha256((STUDIES / name).read_bytes()) for name in dependencies}
    if any(source_hashes[name] != digest for name, digest in old["source_sha256"].items()):
        raise ValueError("batch 35 source dependency changed")
    report = {
        "complete": False, "source": str(args.source.resolve()), "inputs_sha256": old["inputs_sha256"],
        "batch35_report_sha256": sha256((args.source / "report.json").read_bytes()), "source_sha256": source_hashes,
        "python": sys.version, "numpy": np.__version__, "zlib": zlib.ZLIB_RUNTIME_VERSION, "psutil": psutil.__version__,
        "priority": int(process.nice()), "cpu_affinity": process.cpu_affinity(), "data_workers": 0,
        "budgets": BUDGETS, "bootstrap_seed": SEED, "bootstrap_repetitions": RESAMPLES,
    }
    json_save(args.out / "run-config.json", report)
    prepared = {layout: prepare_layout(args.out / layout, data, manifest, layout, args.source / layout)
                for layout in LAYOUTS}
    json_save(args.out / "selected-before-test.json", {layout: entry[1] for layout, entry in prepared.items()})
    results = {}
    for layout in LAYOUTS:
        results[layout] = measure_layout(args.out / layout, data, manifest, layout, args.source / layout,
                                         old["results"][layout]["files"], prepared[layout])
        json_save(args.out / layout / "summary.json", results[layout])
    memory = process.memory_info()
    report.update(complete=True, results=results, seconds=time.perf_counter() - started,
                  peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss))
    json_save(args.out / "report.json", report)
    print(f"Complete in {report['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
