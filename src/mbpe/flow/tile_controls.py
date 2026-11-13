"""Three comparisons on one manifest grid. merge_pair's selection is not edited."""

import json
import math
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from ..structure.bpe.compression_metrics import CompressionMetrics
from ..structure.bpe.controls import GridMerger, encode_skipping_unknown, first_fork, merge_sequence, patch_order, sequence_leaves
from ..structure.config import DEFAULT_EXPERIMENT_SEEDS, TokenizerConfig
from ..structure.synthetic import generate_grids, select_grids
from ..structure.bpe.tokenizer import Tokenizer

INITIAL_ALPHABET = 256


DIAGNOSE_MAX_SYMBOLS = 280


def execute_diagnose(output, seeds=DEFAULT_EXPERIMENT_SEEDS, manifest_version=2, grid_name="tile"):
    """Record the first count-versus-spatial disagreement at a fixed symbol cap."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    limit = DIAGNOSE_MAX_SYMBOLS - INITIAL_ALPHABET
    per_seed = []
    for seed in seeds:
        grid = select_grids(generate_grids(seed, manifest_version), grid_name)
        if len(grid) != 1:
            raise ValueError("diagnose expects one grid")
        grid = grid[0]
        height, width = grid["shape"][1], grid["shape"][2]
        rows = [list(grid["values"][row * width:(row + 1) * width]) for row in range(height)]
        result = first_fork(rows, limit)
        fork = result["fork"]
        per_seed.append({
            "seed": seed,
            "shape": list(grid["shape"]),
            "fork_step": None if fork is None else fork["step"],
            "token_count_at_fork": None if fork is None else fork["token_count"],
            "spatial_token_count_at_fork": None if fork is None else fork["spatial_token_count"],
            "count_pair": None if fork is None else _pack(fork["count_pair"]),
            "spatial_pair": None if fork is None else _pack(fork["spatial_pair"]),
            "count_annotation": None if fork is None else fork["count_annotation"],
            "spatial_annotation": None if fork is None else fork["spatial_annotation"],
            "count_final_tokens": result["count_final_tokens"],
            "spatial_final_tokens": result["spatial_final_tokens"],
            "count_lossless": result["count_lossless"],
            "spatial_lossless": result["spatial_lossless"],
        })
    summary = {
        "manifest_version": manifest_version,
        "grid_name": grid_name,
        "max_symbols": DIAGNOSE_MAX_SYMBOLS,
        "new_symbol_limit": limit,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "note": (
            "诊断固定符号上限 280。分叉时的 token 数是选出这一对之前还剩下的块数。"
            "成员标原始或合成。方向只看这一对当前的相邻方向：全是水平为水平，全是垂直为垂直，两种都有为深层。"
            "次数得分是出现次数，空间项得分是次数占比加 0.3 乘空间一致性。不调用 merge_pair。"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def _pack(value):
    if isinstance(value, tuple):
        return [_pack(item) for item in value]
    return value


def execute_controls(output, seeds=DEFAULT_EXPERIMENT_SEEDS, manifest_version=2,
                     grid_name="tile", max_symbols=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    limit = None if max_symbols is None else max(0, max_symbols - INITIAL_ALPHABET)
    per_seed = [ _one_seed(seed, manifest_version, grid_name, max_symbols, limit) for seed in seeds]
    summary = {
        "manifest_version": manifest_version,
        "grid_name": grid_name,
        "max_symbols": max_symbols,
        "new_symbol_limit": limit,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "mean": _mean(per_seed),
        "note": (
            "停止对照使用训练后的分词器：停止是 encode，跳过是最大对不在词表时改选下一个已知对。"
            "形状和空间对照不调用 merge_pair。符号上限换算成初始 256 之外还能新增的符号数，"
            "达到后不再开始下一轮合并。"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def _one_seed(seed, manifest_version, grid_name, max_symbols, limit):
    grid = select_grids(generate_grids(seed, manifest_version), grid_name)
    if len(grid) != 1:
        raise ValueError("controls expect one grid")
    grid = grid[0]
    height, width = grid["shape"][1], grid["shape"][2]
    rows = [list(grid["values"][row * width:(row + 1) * width]) for row in range(height)]
    return {
        "seed": seed,
        "shape": list(grid["shape"]),
        "contrasts": {
            "stop_vs_skip": _stop_skip(grid, max_symbols),
            "unshuffle_vs_constellation": _shape(rows, limit),
            "count_vs_spatial": _spatial(rows, limit),
        },
    }


def _stop_skip(grid, max_symbols):
    config = TokenizerConfig(seed=0, max_symbols=max_symbols)
    kwargs = config.tokenizer_kwargs()
    kwargs["seed"] = None
    tensor = torch.tensor(grid["values"], dtype=torch.uint8, device="cpu").reshape((1, *grid["shape"]))
    baseline = Tokenizer(**kwargs)
    trained = Tokenizer(**kwargs)
    trained.train(TensorDataset(tensor), 0)
    before = baseline.encode(tensor)
    stopped = trained.encode(tensor)
    skipped = encode_skipping_unknown(trained, tensor)
    metrics = CompressionMetrics(trained)
    return {
        "stop": _tokenizer_arm(tensor, before, stopped, trained, metrics),
        "skip": _tokenizer_arm(tensor, before, skipped, trained, metrics, decode=trained.decode),
    }


def _tokenizer_arm(tensor, before, after, tokenizer, metrics, decode=None):
    if decode is None:
        decode = tokenizer.decode
    try:
        lossless = bool(torch.equal(decode(after), tensor))
        error = None
    except (RuntimeError, ValueError) as exc:
        lossless = False
        error = str(exc)
    ratio = metrics.compute_compression_ratio(tensor, after.sequences[0])
    return {
        "baseline_token_count": len(before.sequences[0]),
        "token_count": len(after.sequences[0]),
        "token_count_delta": len(after.sequences[0]) - len(before.sequences[0]),
        "lossless": lossless,
        "compression_ratio": ratio["compression_ratio"],
        "vocab_size": len(tokenizer.vocab),
        "error": error,
    }


def _shape(rows, limit):
    height, width = len(rows), len(rows[0])
    order = patch_order(height, width)
    ordered = [rows[index // width][index % width] for index in order]
    tokens, _chosen, symbols = merge_sequence(ordered, "count", limit)
    leaves = []
    for token in tokens:
        leaves.extend(sequence_leaves(token["symbol"]))
    constellation = GridMerger(rows, "count", limit)
    return {
        "unshuffle": _grid_arm(len(ordered), len(tokens), 256 + len(symbols), leaves == ordered),
        "constellation": _grid_arm(
            height * width,
            len(constellation.tokens),
            256 + len(constellation.merge_symbols),
            constellation.values() == rows,
        ),
    }


def _spatial(rows, limit):
    height, width = len(rows), len(rows[0])
    by_count = GridMerger(rows, "count", limit)
    by_spatial = GridMerger(rows, "priority", limit)
    return {
        "count": _grid_arm(height * width, len(by_count.tokens), 256 + len(by_count.merge_symbols),
                           by_count.values() == rows),
        "spatial": _grid_arm(height * width, len(by_spatial.tokens), 256 + len(by_spatial.merge_symbols),
                             by_spatial.values() == rows),
    }


def _grid_arm(baseline, token_count, vocab_size, lossless):
    bits = math.ceil(math.log2(vocab_size)) if vocab_size > 1 else 1
    encoded_bits = token_count * bits
    return {
        "baseline_token_count": baseline,
        "token_count": token_count,
        "token_count_delta": token_count - baseline,
        "lossless": lossless,
        "compression_ratio": (baseline * 8) / encoded_bits if encoded_bits else 0,
        "vocab_size": vocab_size,
        "error": None,
    }


def _mean(rows):
    names = rows[0]["contrasts"]
    return {
        contrast: {
            arm: {
                "baseline_token_count": _average(item["contrasts"][contrast][arm]["baseline_token_count"] for item in rows),
                "token_count": _average(item["contrasts"][contrast][arm]["token_count"] for item in rows),
                "token_count_delta": _average(item["contrasts"][contrast][arm]["token_count_delta"] for item in rows),
                "compression_ratio": _average(item["contrasts"][contrast][arm]["compression_ratio"] for item in rows),
                "lossless": all(item["contrasts"][contrast][arm]["lossless"] for item in rows),
            }
            for arm in names[contrast]
        }
        for contrast in names
    }


def _average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0
