"""Version 1 JSON for a tokenizer. Standard library only."""

import json
from pathlib import Path

from .tokenizer import Tokenizer

FORMAT = "mbpe-tokenizer"
VERSION = 1


def save_tokenizer(path, tokenizer):
    text = json.dumps(_payload(tokenizer), ensure_ascii=False, indent=2)
    Path(path).write_text(text + "\n", encoding="utf-8")


def load_tokenizer(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    version = document.get("version") if isinstance(document, dict) else None
    if document.get("format") != FORMAT or version != VERSION:
        raise ValueError(f"unknown tokenizer file version: {version!r}")
    config = _read_config(document.get("config"))
    vocabulary = _read_vocabulary(document.get("vocabulary"))
    merge_order = _read_merge_order(document.get("merge_order"), vocabulary)
    frequency = _read_frequency(document.get("frequency"))
    tokenizer = Tokenizer(
        config["min_freq"],
        config["max_codeword_size"],
        batch_size=config["batch_size"],
        max_workers=config["max_workers"],
        max_symbols=config["max_symbols"],
        seed=config["seed"],
    )
    _install_vocabulary(tokenizer, vocabulary)
    tokenizer.merge_order = merge_order
    tokenizer.freq_counter.total_count = frequency["total_count"]
    tokenizer.freq_counter.count_table = frequency["counts"]
    return tokenizer


def _payload(tokenizer):
    return {
        "format": FORMAT,
        "version": VERSION,
        "config": {
            "max_codeword_size": list(tokenizer.codeword_sizes[0]),
            "batch_size": tokenizer.batch_size,
            "max_workers": tokenizer.max_workers,
            "min_freq": {
                "root": tokenizer.freq_counter.min_freq["root"],
                "merge": tokenizer.freq_counter.min_freq["merge"],
                "freq_counter": tokenizer.freq_counter.min_freq["freq_counter"],
            },
            "max_symbols": tokenizer.max_symbols,
            "seed": tokenizer.seed,
        },
        "merge_order": [
            {
                "pair": _pack(pair),
                "codeword": codeword,
                "codeword_size": list(size),
            }
            for pair, codeword, size in tokenizer.merge_order
        ],
        "vocabulary": [
            {
                "symbol": _pack(symbol),
                "codeword": codeword,
                "codeword_size": list(_codeword_size(tokenizer.vocab, symbol)),
            }
            for symbol, codeword in tokenizer.vocab.symbol2codeword.items()
        ],
        "frequency": {
            "total_count": tokenizer.freq_counter.total_count,
            "counts": [
                {"symbol": _pack(symbol), "count": count}
                for symbol, count in tokenizer.freq_counter.count_table.items()
            ],
        },
    }


def _codeword_size(vocab, symbol):
    for size, symbols in vocab.codeword_size_symbols.items():
        if symbol in symbols:
            return tuple(size)
    raise ValueError(f"symbol has no codeword size: {symbol!r}")


def _pack(symbol):
    if isinstance(symbol, tuple):
        return {"type": "tuple", "items": [_pack(item) for item in symbol]}
    if isinstance(symbol, str):
        return {"type": "str", "value": symbol}
    if isinstance(symbol, bool) or not isinstance(symbol, int):
        try:
            value = int(symbol)
        except (TypeError, ValueError) as error:
            raise TypeError(f"cannot store symbol {symbol!r}") from error
        if not isinstance(symbol, int) and value != symbol:
            raise TypeError(f"cannot store symbol {symbol!r}")
    else:
        value = symbol
    return {"type": "int", "value": value}


def _unpack(packed):
    if not isinstance(packed, dict) or "type" not in packed:
        raise ValueError("symbol tag is missing")
    kind = packed["type"]
    if kind == "tuple":
        items = packed.get("items")
        if not isinstance(items, list):
            raise ValueError("tuple symbol is missing items")
        return tuple(_unpack(item) for item in items)
    if kind == "str":
        if not isinstance(packed.get("value"), str):
            raise ValueError("string symbol is missing a string value")
        return packed["value"]
    if kind == "int":
        value = packed.get("value")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("integer symbol is missing an integer value")
        return value
    raise ValueError(f"unknown symbol type: {kind!r}")


def _read_config(config):
    if not isinstance(config, dict):
        raise ValueError("tokenizer config is missing")
    size = config.get("max_codeword_size")
    batch_size = config.get("batch_size")
    thresholds = config.get("min_freq")
    if (
        not isinstance(size, list)
        or not size
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in size)
    ):
        raise ValueError("max_codeword_size must contain positive integers")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(thresholds, dict):
        raise ValueError("frequency thresholds are missing")
    minimum = {}
    for key in ("root", "merge", "freq_counter"):
        value = thresholds.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"frequency threshold {key} is missing")
        minimum[key] = value
    max_workers = config.get("max_workers")
    if max_workers is not None and (isinstance(max_workers, bool) or not isinstance(max_workers, int)):
        raise ValueError("max_workers must be an integer or null")
    if "max_symbols" not in config or config["max_symbols"] is None:
        max_symbols = None
    else:
        max_symbols = config["max_symbols"]
        if isinstance(max_symbols, bool) or not isinstance(max_symbols, int) or max_symbols <= 0:
            raise ValueError("max_symbols must be a positive integer or null")
    if "seed" not in config or config["seed"] is None:
        seed = None
    else:
        seed = config["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer or null")
    return {
        "max_codeword_size": tuple(size),
        "batch_size": batch_size,
        "max_workers": max_workers,
        "min_freq": minimum,
        "max_symbols": max_symbols,
        "seed": seed,
    }


def _read_vocabulary(items):
    if not isinstance(items, list):
        raise ValueError("vocabulary is missing")
    vocabulary = []
    symbols = set()
    codewords = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("vocabulary item is not an object")
        symbol = _unpack(item.get("symbol"))
        codeword = item.get("codeword")
        size = item.get("codeword_size")
        if not isinstance(codeword, str):
            raise ValueError("codeword must be a string")
        if (
            not isinstance(size, list)
            or not size
            or any(isinstance(part, bool) or not isinstance(part, int) or part <= 0 for part in size)
        ):
            raise ValueError("codeword_size must contain positive integers")
        if symbol in symbols or codeword in codewords:
            raise ValueError("duplicate symbol or codeword")
        symbols.add(symbol)
        codewords.add(codeword)
        vocabulary.append((symbol, codeword, tuple(size)))
    return vocabulary


def _read_merge_order(items, vocabulary):
    if not isinstance(items, list):
        raise ValueError("merge order is missing")
    known = {symbol: (codeword, size) for symbol, codeword, size in vocabulary}
    order = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("merge order item is not an object")
        pair = _unpack(item.get("pair"))
        codeword = item.get("codeword")
        size = item.get("codeword_size")
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("merge pair must contain two symbols")
        if not isinstance(codeword, str):
            raise ValueError("merge codeword must be a string")
        if not isinstance(size, list):
            raise ValueError("merge codeword_size is missing")
        size = tuple(size)
        recorded = known.get(pair)
        if recorded != (codeword, size):
            raise ValueError("merge order does not match the vocabulary")
        order.append((pair, codeword, size))
    return order


def _read_frequency(frequency):
    if not isinstance(frequency, dict):
        raise ValueError("frequency table is missing")
    total = frequency.get("total_count")
    counts = frequency.get("counts")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total_count must be a non-negative integer")
    if not isinstance(counts, list):
        raise ValueError("frequency counts are missing")
    table = {}
    for item in counts:
        if not isinstance(item, dict):
            raise ValueError("frequency count is not an object")
        symbol = _unpack(item.get("symbol"))
        count = item.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("frequency count must be a non-negative integer")
        if symbol in table:
            raise ValueError("duplicate frequency symbol")
        table[symbol] = count
    return {"total_count": total, "counts": table}


def _install_vocabulary(tokenizer, vocabulary):
    existing = dict(tokenizer.vocab.symbol2codeword)
    for symbol, codeword, size in vocabulary:
        current = existing.get(symbol)
        if current is None:
            tokenizer.vocab.add(symbol, codeword, size)
            continue
        if current != codeword or _codeword_size(tokenizer.vocab, symbol) != size:
            raise ValueError("vocabulary conflicts with the initial alphabet")
    saved = [(symbol, codeword) for symbol, codeword, _size in vocabulary]
    if list(tokenizer.vocab.symbol2codeword.items()) != saved:
        raise ValueError("vocabulary does not match the saved order")
