"""Shared learning and independent archive checks for the integer-grid studies."""

from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

from batch30_mask_baselines import learn_apply
from batch33_celeba_files import json_save, sha256
from batch34_token_coding import apply_prefix
import grid_bpe_file as archive
from lossless_nd_candidate import LosslessNDCandidate
import numpy as np
import psutil


def constrain_cpu():
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:1])
    if process.nice() != psutil.IDLE_PRIORITY_CLASS:
        raise RuntimeError("Idle priority is required for the study")
    return process


def domain_of(shape, cells):
    from math import prod

    return None if len(cells) == prod(shape) else tuple(sorted(cells))


def learn_linear(samples, base, mode, budget):
    ids = {value: i for i, value in enumerate(base)}
    originals = [np.asarray([ids[cells[point]] for point in archive.scan_points(shape, domain_of(shape, cells), mode)],
                            dtype=np.int32) for shape, cells in samples]
    rules, _ = learn_apply(originals, {"unused": []}, len(base), budget, set())
    return tuple((left, right) for left, right, _ in rules)


def linear_packets(spec, samples):
    from bpe_block_file import FileSpec

    base = {value: i for i, value in enumerate(spec.base)}
    originals, metadata = [], []
    for shape, cells in samples:
        domain = domain_of(shape, cells)
        literals = tuple(sorted(set(cells.values()) - base.keys()))
        ids = base | dict(zip(literals, range(spec.symbols(0), spec.symbols(0) + len(literals))))
        originals.append([ids[cells[point]] for point in archive.scan_points(shape, domain, spec.mode)])
        metadata.append((shape, domain, literals))
    adapter = FileSpec("row", 1, len(spec.base), spec.grammars[0])
    return [archive.Packet(*parts, tuple(tokens)) for parts, tokens in zip(metadata, apply_prefix(adapter, originals))]


def nd_spec(model, budget):
    rules = tuple((a, b, *delta) for (a, b, delta), _ in model.rules[:budget])
    return archive.Spec("nd", model.rank, tuple(model.base), (rules,))


def nd_packets(spec, samples):
    model = archive.nd_decoder(spec)
    return [archive.packet_from_nd(model.encode(shape, cells)) for shape, cells in samples]


def learn_nd(samples, budget):
    model = LosslessNDCandidate(len(samples[0][0]))
    model.learn(samples, budget=budget, score="frequency")
    return model


def reference_bytes(samples):
    return b"".join(archive.grid_bytes(shape, domain_of(shape, cells), [cells[point] for point in sorted(cells)])
                    for shape, cells in samples)


def measure(spec, directions, packets, codec, lengths=None, width=0):
    buffer = BytesIO()
    return archive.write_archive(buffer, spec, directions, packets, codec, lengths, width)


def write_checked(path, spec, directions, packets, codec, lengths, reference, width=0):
    with path.open("xb") as handle:
        cost = archive.write_archive(handle, spec, directions, packets, codec, lengths, width)
    with path.open("rb") as handle:
        read_back = archive.read_archive(handle)
    if read_back != (spec, directions, packets, codec, lengths, width, cost) or path.stat().st_size != cost["total_bytes"]:
        raise ValueError(f"archive accounting or packet differs: {path}")

    decoded = path.with_suffix(".decoded")
    decoder = Path(archive.__file__).resolve()
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(decoder), str(path), str(decoded)],
                            capture_output=True, encoding="utf-8", check=False)
    if result.returncode:
        json_save(path.with_suffix(".failure.json"), {"exit_code": result.returncode, "error": result.stderr})
        raise ValueError(f"independent decoder failed: {path}")
    receipt = json.loads(result.stdout)
    process = psutil.Process()
    if receipt != {"cost": cost, "cpu": process.cpu_affinity(), "priority": int(psutil.IDLE_PRIORITY_CLASS)}:
        raise ValueError(f"independent decoder accounting or CPU limits differ: {path}")
    recovered = decoded.read_bytes()
    if recovered != reference:
        json_save(path.with_suffix(".failure.json"), {"decoded": sha256(recovered), "expected": sha256(reference)})
        raise ValueError(f"independent decoder changed grid shape, domain or values: {path}")
    return {**cost, "file": path.name, "sha256": sha256(path.read_bytes()), "decoded_sha256": sha256(recovered),
            "lossless_grids": len(packets), "rules": [len(rules) for rules in spec.grammars], "mode": spec.mode}
