"""Recode frozen batch 33 token streams with canonical Huffman codes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import zlib

from batch30_mask_baselines import merge_one
from batch33_celeba_files import SEED, STUDIES, compare_files, json_save, sha256
from bpe_block_file import HUFFMAN_VERSION, FileSpec, read_token_archive, write_archive, write_huffman_archive
from canonical_huffman import lengths_from_validation
import numpy as np
import psutil


CASES = {
    "4level": {"selected": ("2d-256", "hilbert-256")},
    "bin128": {"selected": ("2d-64", "hilbert-256"), "same_budget": ("2d-64", "hilbert-64")},
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_frozen(directory, name, expected):
    path = directory / f"{name}.mbpe"
    blob = path.read_bytes()
    if len(blob) != expected["total_bytes"] or sha256(blob) != expected["file_sha256"]:
        raise ValueError(f"batch 33 archive changed: {path}")
    with path.open("rb") as handle:
        spec, sequences, cost = read_token_archive(handle)
    if any(expected[key] != value for key, value in cost.items()):
        raise ValueError(f"batch 33 accounting changed: {path}")
    cost["file_sha256"] = expected["file_sha256"]
    return spec, sequences, cost


def check_decoded_grids(blob, manifest, mask, split, side):
    size = side * side
    samples = manifest["splits"][split]
    if len(blob) != len(samples) * size:
        raise ValueError("decoded grid length differs from the batch 33 input")
    return [
        index for index, image in enumerate(samples)
        if sha256(blob[index * size:(index + 1) * size]) != image["grid_sha256"][mask]
    ]


def write_checked(directory, name, writer, spec, sequences, reference, manifest, mask, split):
    archive = directory / f"{name}.mbpe"
    decoded = directory / f"{name}.decoded"
    started = time.perf_counter()
    with archive.open("xb") as handle:
        cost = writer(handle)
    written = time.perf_counter() - started
    with archive.open("rb") as handle:
        found_spec, found_tokens, found_cost = read_token_archive(handle)
    if (found_spec, found_tokens, found_cost) != (spec, sequences, cost):
        raise ValueError(f"token stream or accounting mismatch: {archive}")
    if archive.stat().st_size != cost["total_bytes"]:
        raise ValueError(f"file size mismatch: {archive}")
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", str(STUDIES / "bpe_block_file.py"), str(archive), str(decoded)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    decoder_seconds = time.perf_counter() - started
    cost.update(file=archive.name, file_sha256=sha256(archive.read_bytes()),
                write_seconds=written, decode_seconds=decoder_seconds,
                tokens=sum(cost["token_counts"]), bits_per_cell=8 * cost["total_bytes"] / (len(sequences) * spec.cells))
    result_path = directory / f"{name}.json"
    if proc.returncode:
        cost.update(decoder_error=proc.stderr, decoder_exit_code=proc.returncode)
        json_save(result_path, cost)
        raise ValueError(f"independent decoder failed: {archive}")
    decoder = json.loads(proc.stdout)
    if any(decoder[key] != value for key, value in found_cost.items()):
        raise ValueError(f"independent decoder accounting mismatch: {archive}")
    if decoder["priority"] != int(psutil.IDLE_PRIORITY_CLASS) or len(decoder["cpu_affinity"]) != 1:
        raise ValueError("decoder did not inherit the run's CPU constraints")
    recovered = decoded.read_bytes()
    cost["decoded_sha256"] = sha256(recovered)
    cost["lossless_bad"] = check_decoded_grids(recovered, manifest, mask, split, spec.side)
    cost["decoder_peak_working_set_bytes"] = decoder["peak_working_set_bytes"]
    json_save(result_path, cost)
    if recovered != reference or cost["lossless_bad"]:
        raise ValueError(f"decoded grid differs from the frozen input: {archive}")
    print(f"{mask}/{split} {name}: {cost['total_bytes']} bytes, {len(sequences)} grids restored", flush=True)
    return cost


def apply_prefix(spec, originals):
    sequences = []
    for original in originals:
        sequence = np.asarray(original, dtype=np.int32)
        for token, (left, right) in enumerate(spec.rules, start=spec.alphabet):
            sequence = merge_one(sequence, left, right, token)
        sequences.append(sequence.tolist())
    return sequences


def difference_of_gaps(old_2d, old_1d, new_2d, new_1d):
    old_gap = old_2d["total_bytes"] - old_1d["total_bytes"]
    new_gap = new_2d["total_bytes"] - new_1d["total_bytes"]
    payloads = [np.asarray(row["payload_sizes"], dtype=np.int64) for row in (old_2d, old_1d, new_2d, new_1d)]
    per_image = (payloads[2] - payloads[3]) - (payloads[0] - payloads[1])
    fixed = new_gap - old_gap - int(per_image.sum())
    indices = np.random.default_rng(SEED).integers(0, len(per_image), size=(10_000, len(per_image)))
    interval = np.quantile(fixed + per_image[indices].sum(axis=1), [0.025, 0.975])
    return {
        "zlib_total_gap_bytes": old_gap, "huffman_total_gap_bytes": new_gap,
        "gap_change_bytes": new_gap - old_gap,
        "gap_change_interval_bytes": interval.tolist(),
        "gap_shrinkage": 1 - new_gap / old_gap if old_gap > 0 else None,
    }


def run_mask(source, output, mask, original, manifest):
    directory = output / mask
    directory.mkdir()
    validation_dir, test_dir = directory / "validation", directory / "test"
    validation_dir.mkdir()
    test_dir.mkdir()
    result = {"huffman": {"validation": {}, "test": {}}, "zlib_test": {}, "comparisons": {}}
    names = tuple(dict.fromkeys(name for pair in CASES[mask].values() for name in pair))
    references = {}
    for split in ("validation", "test"):
        raw_row = original[split]["raw"]
        blob = (source / mask / split / "raw.decoded").read_bytes()
        if sha256(blob) != raw_row["decoded_sha256"] or check_decoded_grids(blob, manifest, mask, split, 128):
            raise ValueError(f"invalid batch 33 {mask}/{split} reference")
        references[split] = blob
    for name in ("raw", "hilbert-0", "2d-0"):
        spec, sequences, cost = read_frozen(source / mask / "test", name, original["test"][name])
        result["zlib_test"][name] = cost

    for name in names:
        spec, validation, _ = read_frozen(source / mask / "validation", name, original["validation"][name])
        lengths = lengths_from_validation(validation, spec.symbols)
        result["huffman"]["validation"][name] = write_checked(
            validation_dir, name, lambda handle: write_huffman_archive(handle, spec, validation, lengths),
            spec, validation, references["validation"], manifest, mask, "validation",
        )
        if name in original["test"]:
            test_spec, test, cost = read_frozen(source / mask / "test", name, original["test"][name])
            if test_spec != spec:
                raise ValueError(f"validation and test grammars differ: {name}")
        else:
            if mask != "bin128" or name != "hilbert-64":
                raise ValueError(f"unexpected missing test stream: {mask}/{name}")
            v0, validation_zero, _ = read_frozen(
                source / mask / "validation", "hilbert-0", original["validation"]["hilbert-0"],
            )
            t0, test_zero, _ = read_frozen(source / mask / "test", "hilbert-0", original["test"]["hilbert-0"])
            full_spec, _, _ = read_frozen(
                source / mask / "test", "hilbert-256", original["test"]["hilbert-256"],
            )
            if v0 != t0 or FileSpec("hilbert", spec.side, spec.alphabet, full_spec.rules[:64]) != spec:
                raise ValueError("Hilbert 64 rules are not the frozen prefix")
            if apply_prefix(spec, validation_zero) != validation:
                raise ValueError("Hilbert 64 replay disagrees with the saved validation stream")
            test = apply_prefix(spec, test_zero)
            cost = write_checked(
                test_dir, "hilbert-64-zlib", lambda handle: write_archive(handle, spec, test),
                spec, test, references["test"], manifest, mask, "test",
            )
        result["zlib_test"][name] = cost
        result["huffman"]["test"][name] = write_checked(
            test_dir, name, lambda handle: write_huffman_archive(handle, spec, test, lengths),
            spec, test, references["test"], manifest, mask, "test",
        )

    zlib_rows = result["zlib_test"]
    huffman_rows = result["huffman"]["test"]
    for case, (nd, linear) in CASES[mask].items():
        result["comparisons"][case] = {
            "methods": [nd, linear],
            "zlib": compare_files(zlib_rows[nd], zlib_rows[linear]),
            "huffman": compare_files(huffman_rows[nd], huffman_rows[linear]),
            "codec_change": difference_of_gaps(zlib_rows[nd], zlib_rows[linear], huffman_rows[nd], huffman_rows[linear]),
        }
    json_save(directory / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:1])
    started = time.perf_counter()
    report_path = args.source / "report.json"
    manifest_path = args.source / "inputs.json"
    original, manifest = read_json(report_path), read_json(manifest_path)
    if not original["complete"] or original["file_version"] != 1 or original["zlib"] != zlib.ZLIB_RUNTIME_VERSION:
        raise ValueError("expected the completed batch 33 v1 run with matching zlib")
    if sha256(manifest_path.read_bytes()) != original["inputs_sha256"]:
        raise ValueError("batch 33 input manifest changed")
    for mask, pairs in CASES.items():
        selected = original["results"][mask]["selected_learned_only"]
        if pairs["selected"] != (selected["2d"], selected["hilbert"]):
            raise ValueError(f"batch 33 selected grammar differs for {mask}")
    files = ("batch34_token_coding.py", "bpe_block_file.py", "canonical_huffman.py",
             *[name for name in original["source_sha256"] if name != "bpe_block_file.py"])
    hashes = {name: sha256((STUDIES / name).read_bytes()) for name in files}
    for name, digest in original["source_sha256"].items():
        if name != "bpe_block_file.py" and hashes[name] != digest:
            raise ValueError(f"batch 33 dependency changed: {name}")
    report = {
        "source": str(args.source.resolve()), "batch33_report_sha256": sha256(report_path.read_bytes()),
        "inputs_sha256": original["inputs_sha256"], "source_sha256": hashes, "python": sys.version,
        "numpy": np.__version__, "psutil": psutil.__version__, "zlib": zlib.ZLIB_RUNTIME_VERSION,
        "file_version": HUFFMAN_VERSION, "priority": int(process.nice()), "cpu_affinity": process.cpu_affinity(),
        "data_workers": 0, "seed": SEED, "cases": CASES, "complete": False,
    }
    json_save(args.out / "run-config.json", report)
    report["results"] = {
        mask: run_mask(args.source, args.out, mask, original["results"][mask], manifest) for mask in CASES
    }
    memory = process.memory_info()
    report.update(complete=True, seconds=time.perf_counter() - started,
                  peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss))
    json_save(args.out / "report.json", report)
    print(f"Complete in {report['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
