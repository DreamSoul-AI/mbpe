"""Deterministic synthetic grids. Parameters live in the versioned manifest file."""

import json
import random
from pathlib import Path

MANIFEST_VERSION = 1
_MANIFEST = Path(__file__).with_name("synthetic_manifest_v1.json")


def load_manifest(version=MANIFEST_VERSION):
    if version == 1:
        return _load_v1()
    if version == 2:
        return _load_v2()
    raise ValueError(f"unknown synthetic manifest version: {version}")


def select_grids(grids, grid_name=None):
    if grid_name is None:
        return list(grids)
    selected = [grid for grid in grids if grid.get("name") == grid_name]
    if not selected:
        raise ValueError(f"unknown grid name: {grid_name}")
    return selected


def generate_grids(seed, version=MANIFEST_VERSION):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if version == 1:
        return _generate_v1(seed)
    if version == 2:
        return _generate_v2(seed)
    raise ValueError(f"unknown synthetic manifest version: {version}")


def _load_v1():
    document = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    if document.get("manifest_version") != 1:
        raise ValueError(f"unknown synthetic manifest version: {document.get('manifest_version')!r}")
    shapes = tuple(tuple(shape) for shape in document["shapes"])
    if not shapes or any(not shape or any(type(dim) is not int or dim <= 0 for dim in shape) for shape in shapes):
        raise ValueError("synthetic manifest shapes must be positive integers")
    low, high = document["value_min"], document["value_max"]
    if type(low) is not int or type(high) is not int or not 0 <= low <= high <= 255:
        raise ValueError("synthetic values must be integers in 0..255")
    return {"manifest_version": 1, "rule": document["rule"], "shapes": shapes,
            "value_min": low, "value_max": high}


def _generate_v1(seed):
    spec = _load_v1()
    generator = random.Random(seed)
    grids = []
    for shape in spec["shapes"]:
        count = 1
        for dim in shape:
            count *= dim
        values = [generator.randint(spec["value_min"], spec["value_max"]) for _ in range(count)]
        grids.append({"shape": shape, "values": tuple(values)})
    return grids


def _load_v2():
    document = json.loads(Path(__file__).with_name("synthetic_manifest_v2.json").read_text(encoding="utf-8"))
    if document.get("manifest_version") != 2:
        raise ValueError(f"unknown synthetic manifest version: {document.get('manifest_version')!r}")
    grids = []
    for item in document["grids"]:
        shape = tuple(item["shape"])
        if len(shape) != 3 or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError("version 2 shape must be three positive integers")
        if shape[1] % 2 or shape[2] % 2:
            raise ValueError("version 2 height and width must be divisible by 2")
        low, high = item["value_min"], item["value_max"]
        if type(low) is not int or type(high) is not int or not 0 <= low <= high <= 255:
            raise ValueError("synthetic values must be integers in 0..255")
        pattern = item["pattern"]
        spec = {"name": item["name"], "pattern": pattern, "shape": shape, "value_min": low, "value_max": high}
        if pattern == "stripe":
            period = item["period"]
            if type(period) is not int or period <= 0:
                raise ValueError("stripe period must be a positive integer")
            spec["period"] = period
        elif pattern == "tile":
            tile_height, tile_width = item["tile_height"], item["tile_width"]
            if type(tile_height) is not int or type(tile_width) is not int or tile_height <= 0 or tile_width <= 0:
                raise ValueError("tile size must be positive integers")
            if shape[1] % tile_height or shape[2] % tile_width:
                raise ValueError("tile size must divide the grid")
            spec["tile_height"] = tile_height
            spec["tile_width"] = tile_width
        elif pattern != "noise":
            raise ValueError(f"unknown synthetic pattern: {pattern!r}")
        grids.append(spec)
    if not grids:
        raise ValueError("version 2 manifest has no grids")
    return {"manifest_version": 2, "rule": document["rule"], "grids": tuple(grids)}


def _generate_v2(seed):
    spec = _load_v2()
    generator = random.Random(seed)
    grids = []
    for item in spec["grids"]:
        shape = item["shape"]
        low, high = item["value_min"], item["value_max"]
        span = high - low + 1
        if item["pattern"] == "stripe":
            phase = generator.randint(0, span - 1)
            period = item["period"]
            values = _fill(shape, lambda _y, x: low + (x // period + phase) % span)
        elif item["pattern"] == "tile":
            tile_height, tile_width = item["tile_height"], item["tile_width"]
            tile = [generator.randint(low, high) for _ in range(tile_height * tile_width)]
            values = _fill(shape, lambda y, x: tile[(y % tile_height) * tile_width + (x % tile_width)])
        else:
            values = _fill(shape, lambda _y, _x: generator.randint(low, high))
        grids.append({"name": item["name"], "shape": shape, "values": values})
    return grids


def _fill(shape, value_at):
    _channels, height, width = shape
    values = []
    for _channel in range(shape[0]):
        for y in range(height):
            for x in range(width):
                values.append(value_at(y, x))
    return tuple(values)
