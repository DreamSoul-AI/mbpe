"""Train and encode the versioned synthetic grids. One tokenizer per seed."""

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from ..structure.bpe.compression_metrics import CompressionMetrics
from ..structure.bpe.persist import save_tokenizer
from ..structure.config import DEFAULT_EXPERIMENT_SEEDS, TokenizerConfig
from ..structure.synthetic import MANIFEST_VERSION, generate_grids, load_manifest, select_grids
from ..structure.bpe.tokenizer import Tokenizer


def default_output_directory():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / f"{stamp}-manifest-v{MANIFEST_VERSION}"


def execute(output, seeds=DEFAULT_EXPERIMENT_SEEDS, manifest_version=MANIFEST_VERSION,
            max_codeword_size=(1, 2, 2), batch_size=4, max_symbols=None, grid_name=None):
    spec = load_manifest(manifest_version)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for seed in seeds:
        config = TokenizerConfig(
            max_codeword_size=tuple(max_codeword_size),
            batch_size=batch_size,
            seed=seed,
            max_symbols=max_symbols,
        )
        per_seed.append(_run_seed(output, spec, config, grid_name))
    summary = {
        "manifest_version": spec["manifest_version"],
        "manifest_rule": spec["rule"],
        "grid_name": grid_name,
        "seeds": list(seeds),
        "max_symbols": max_symbols,
        "per_seed": per_seed,
        "mean": _mean(per_seed),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    if not summary["mean"]["lossless"]:
        return 1
    return 0


def _run_seed(output, spec, config, grid_name=None):
    grids = select_grids(generate_grids(config.seed, spec["manifest_version"]), grid_name)
    tensors = [_tensor(grid) for grid in grids]
    kwargs = config.tokenizer_kwargs()
    baseline = Tokenizer(**kwargs)
    tokenizer = Tokenizer(**kwargs)
    for tensor in tensors:
        tokenizer.train(TensorDataset(tensor), 0)
    metrics = CompressionMetrics(tokenizer)
    samples = []
    for index, tensor in enumerate(tensors):
        before = baseline.encode(tensor)
        after = tokenizer.encode(tensor)
        restored = tokenizer.decode(after)
        ratio = metrics.compute_compression_ratio(tensor, after.sequences[0])
        samples.append({
            "index": index,
            "shape": list(tensor.shape),
            "baseline_token_count": len(before.sequences[0]),
            "token_count": len(after.sequences[0]),
            "token_count_delta": len(after.sequences[0]) - len(before.sequences[0]),
            "compression_ratio": ratio["compression_ratio"],
            "lossless": bool(torch.equal(restored, tensor)),
        })
    name = f"tokenizer-seed-{config.seed}.json"
    save_tokenizer(output / name, tokenizer)
    return {
        "seed": config.seed,
        "tokenizer": name,
        "vocab_size": len(tokenizer.vocab),
        "max_symbols": config.max_symbols,
        "baseline_token_count": sum(item["baseline_token_count"] for item in samples),
        "token_count": sum(item["token_count"] for item in samples),
        "token_count_delta": sum(item["token_count_delta"] for item in samples),
        "compression_ratio": _average(item["compression_ratio"] for item in samples),
        "lossless": all(item["lossless"] for item in samples),
        "samples": samples,
    }


def _tensor(grid):
    return torch.tensor(grid["values"], dtype=torch.uint8, device="cpu").reshape((1, *grid["shape"]))


def _average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0


def _mean(rows):
    return {
        "baseline_token_count": _average(row["baseline_token_count"] for row in rows),
        "token_count": _average(row["token_count"] for row in rows),
        "token_count_delta": _average(row["token_count_delta"] for row in rows),
        "compression_ratio": _average(row["compression_ratio"] for row in rows),
        "lossless": all(row["lossless"] for row in rows),
    }
