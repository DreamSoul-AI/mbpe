"""Compare raw and left-difference 2D BPE on the frozen batch 38 inputs."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from batch33_celeba_files import json_save, sha256
from batch34_token_coding import read_json
from batch38_native_grids import as_samples, freeze_inputs, unpack_model
from grid_bpe_study import constrain_cpu, learn_nd, nd_spec
from lossless_nd_candidate import LosslessNDCandidate


RULES = 256
EMBEDDING_WIDTH = 192
BYTES_PER_POSITION = EMBEDDING_WIDTH * 2


def left_difference(images):
    residuals = images.copy()
    residuals[..., 1:] = np.subtract(images[..., 1:], images[..., :-1], dtype=np.uint8)
    return residuals


def check_reconstruction(encoder, decoder, images, originals, *, residual):
    lengths = []
    for (shape, cells), original in zip(as_samples(images), originals, strict=True):
        packet = encoder.encode(shape, cells)
        decoded_shape, decoded = decoder.decode(packet)
        if packet["literals"] or decoded_shape != shape or decoded != cells:
            raise ValueError(f"BPE did not reconstruct the expected grid at image {len(lengths)}")
        values = np.fromiter((decoded[point] for point in sorted(decoded)), dtype=np.uint8).reshape(shape)
        if residual:
            values = np.cumsum(values, axis=1, dtype=np.uint8)
        if not np.array_equal(values, original):
            raise ValueError(f"pixel reconstruction failed at image {len(lengths)}")
        lengths.append(len(packet["tokens"]))
    return lengths


def evaluate(images, original, path, *, residual):
    training = as_samples(images["learn"])
    # Reserve every possible input byte without adding any adjacent training pairs.
    missing = sorted(set(range(256)) - set(map(int, np.unique(images["learn"]))))
    training.extend(((1, 1), {(0, 0): value}) for value in missing)
    encoder = learn_nd(training, RULES)
    if len(encoder.rules) != RULES or encoder.base != list(range(256)):
        raise ValueError("expected exactly 256 byte values and 256 merge rules")
    json_save(path, encoder.dictionary())
    decoder = LosslessNDCandidate.from_dictionary(read_json(path))
    return nd_spec(encoder, RULES), {
        split: check_reconstruction(encoder, decoder, images[split], original[split], residual=residual)
        for split in ("validation", "test")
    }, missing


def fixed_pixel_groups(images, size):
    flat = images.reshape(len(images), -1)
    padding = -flat.shape[1] % size
    groups = np.pad(flat, ((0, 0), (0, padding))).reshape(len(images), -1, size)
    ids = np.zeros(groups.shape[:2], dtype=np.uint32)
    for byte in range(size):
        ids = (ids << 8) | groups[..., byte]
    restored = np.empty_like(groups)
    for byte in range(size):
        restored[..., byte] = (ids >> (8 * (size - 1 - byte))) & 255
    if not np.array_equal(restored.reshape(len(images), -1)[:, :flat.shape[1]], flat):
        raise ValueError(f"fixed groups of {size} did not reconstruct the original images")
    return {
        "possible_ids": 256**size,
        "positions_per_image": ids.shape[1],
        "padding_pixels_per_image": padding,
        "activation_bytes_per_image": ids.shape[1] * BYTES_PER_POSITION,
        "lossless_images": len(images),
    }


def dataset_result(args, name, grids, previous):
    directory = args.out / name
    directory.mkdir()
    source = args.batch38 / name / "selected-before-test.json"
    if sha256(source.read_bytes()) != previous["selection_sha256"][name]:
        raise ValueError(f"batch 38 frozen model differs for {name}")
    frozen = read_json(source)["256"]["models"]["nd"]
    frozen_spec, _ = unpack_model(frozen)
    raw_spec, raw, raw_missing = evaluate(grids, grids, directory / "raw-dictionary.json", residual=False)
    if (raw_spec != frozen_spec
            or raw["validation"] != frozen["costs"]["huffman"]["token_counts"]
            or raw["test"] != previous["results"][name]["256"]["files"]["nd"]["huffman"]["token_counts"]):
        raise ValueError(f"raw pixel rules or per-image lengths differ from batch 38 for {name}")

    differences = {split: left_difference(images) for split, images in grids.items()}
    _, residual, residual_missing = evaluate(
        differences, grids, directory / "residual-dictionary.json", residual=True,
    )
    raw_counts = np.asarray(raw["test"], dtype=np.int64)
    residual_counts = np.asarray(residual["test"], dtype=np.int64)
    gains = raw_counts - residual_counts
    pixel_count = int(grids["test"][0].size)
    raw_total, residual_total = sum(raw["test"]), sum(residual["test"])
    result = {
        "images": {split: len(images) for split, images in grids.items()},
        "pixels_per_image": pixel_count,
        "rules_per_method": RULES,
        "raw_missing_training_bytes": raw_missing,
        "residual_missing_training_bytes": residual_missing,
        "per_image_tokens": {"raw": raw, "residual": residual},
        "raw_test_tokens": raw_total,
        "residual_test_tokens": residual_total,
        "residual_fewer_test_tokens_percent": 100 * (raw_total - residual_total) / raw_total,
        "test_shorter": int(np.count_nonzero(gains > 0)),
        "test_same": int(np.count_nonzero(gains == 0)),
        "test_longer": int(np.count_nonzero(gains < 0)),
        "test_with_at_least_half_positions_saved_vs_pixels": int(np.count_nonzero(2 * residual_counts <= pixel_count)),
        "fixed_pixel_groups": {size: fixed_pixel_groups(grids["test"], size) for size in (2, 3, 4)},
    }
    print(f"{name}: {raw_total} -> {residual_total} test tokens; "
          f"{result['residual_fewer_test_tokens_percent']:.2f}% fewer; all images restored", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch38", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    process = constrain_cpu()
    started = perf_counter()
    source = args.batch38 / "inputs.json"
    previous = read_json(args.batch38 / "report.json")
    if not previous["complete"] or sha256(source.read_bytes()) != previous["inputs_sha256"]:
        raise ValueError("expected the completed batch 38 with its frozen inputs")
    data, manifest = freeze_inputs(args.backup)
    if manifest != read_json(source):
        raise ValueError("selected samples differ from batch 38")

    args.out.mkdir(parents=True, exist_ok=False)
    results = {
        name: dataset_result(args, name, {split: np.stack(images) for split, images in grids.items()}, previous)
        for name, grids in data.items()
    }
    json_save(args.out / "report.json", {
        "complete": True,
        "inputs_sha256": previous["inputs_sha256"],
        "source_sha256": sha256(Path(__file__).read_bytes()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "priority": int(process.nice()),
        "cpu_affinity": process.cpu_affinity(),
        "workers": 0,
        "embedding_width": EMBEDDING_WIDTH,
        "embedding_dtype": "BF16",
        "activation_bytes_per_position": BYTES_PER_POSITION,
        "seconds": perf_counter() - started,
        "results": results,
    })


if __name__ == "__main__":
    main()
