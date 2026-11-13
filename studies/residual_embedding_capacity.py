"""Compare byte-valued BF16 slots for pixels and frozen residual BPE."""

import json
from pathlib import Path
from struct import pack, unpack
import sys
from time import perf_counter

import numpy as np
import psutil

from bpe_block_file import pack_symbols, unpack_symbols
from lossless_nd_candidate import LosslessNDCandidate
from native_residual_bpe import left_difference


def embedding_round_trip(blob, capacity, torch):
    positions = (len(blob) + capacity - 1) // capacity
    padded = blob + bytes(positions * capacity - len(blob))
    values = torch.frombuffer(bytearray(padded), dtype=torch.uint8).reshape(positions, capacity)
    embedding = torch.zeros((positions, 192), dtype=torch.bfloat16)
    embedding[:, :capacity] = values
    recovered = embedding[:, :capacity].to(torch.uint8).reshape(-1).numpy().tobytes()
    if recovered[:len(blob)] != blob or any(recovered[len(blob):]):
        raise ValueError("BF16 packing changed bytes")
    return positions, recovered[:len(blob)]


def main():
    process = psutil.Process()
    process.nice(psutil.IDLE_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:1])
    sys.path.insert(0, "src")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = perf_counter()
    root = Path("tmp")
    source = root / "2026-11-14-native-residual-bpe"
    prior = json.loads((source / "report.json").read_bytes())
    manifest = json.loads((root / "2026-11-13-batch38-native-grids/inputs.json").read_bytes())
    from hashlib import sha256

    results = {}
    for dataset in ("cifar", "svhn"):
        model = LosslessNDCandidate.from_dictionary(json.loads((source / dataset / "residual-dictionary.json").read_bytes()))
        if len(model.rules) != 256 or model.base != list(range(256)):
            raise ValueError("unexpected frozen dictionary")
        rows = (root / "2026-11-13-batch38-native-grids" / dataset / "column-0-huffman.decoded").read_bytes().splitlines()
        output = {k: [] for k in (2, 3, 192)}
        for index, (line, reference) in enumerate(zip(rows, manifest[dataset]["test"]["samples"], strict=True)):
            shape, domain, pixels = json.loads(line)
            raw = bytes(pixels)
            if shape != [32, 32] or domain is not None or sha256(raw).hexdigest() != reference["sha256"]:
                raise ValueError("input identity mismatch")
            image = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            residual = left_difference(image)
            cells = {point: int(value) for point, value in np.ndenumerate(residual)}
            packet = model.encode(tuple(shape), cells)
            tokens = packet["tokens"]
            if len(tokens) != prior["results"][dataset]["per_image_tokens"]["residual"]["test"][index] or packet["literals"]:
                raise ValueError("frozen BPE sequence mismatch")
            # Count is part of the input capacity and permits exact padding removal.
            blob = pack("<H", len(tokens)) + pack_symbols(tokens, 9, 512)
            for capacity in output:
                raw_positions, recovered_raw = embedding_round_trip(raw, capacity, torch)
                positions, restored_blob = embedding_round_trip(blob, capacity, torch)
                count, = unpack("<H", restored_blob[:2])
                decoded_tokens = unpack_symbols(restored_blob[2:], count, 9, 512)
                if decoded_tokens != tokens or recovered_raw != raw:
                    raise ValueError("packed token stream changed")
                restored_packet = dict(packet, tokens=decoded_tokens)
                _, restored_cells = model.decode(restored_packet)
                decoded = np.array([restored_cells[point] for point in sorted(restored_cells)], dtype=np.uint8).reshape(shape)
                if not np.array_equal(np.cumsum(decoded, axis=1, dtype=np.uint8), image):
                    raise ValueError("packed BPE pixel restoration failed")
                output[capacity].append({"raw_positions": raw_positions, "bpe_positions": positions})
        results[dataset] = output
        for capacity, records in output.items():
            a = sum(row["raw_positions"] for row in records)
            b = sum(row["bpe_positions"] for row in records)
            print(dataset, capacity, a / len(rows), b / len(rows), 100 * (1 - b/a), flush=True)
    out = root / "2026-11-14-residual-embedding-capacity"
    out.mkdir(exist_ok=False)
    report = {"width": 192, "dtype": "BF16", "slot_bits": 8, "bpe_bits": 9,
              "bpe_count_bytes": 2, "priority": int(process.nice()), "affinity": process.cpu_affinity(),
              "seconds": perf_counter() - started, "results": results}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
