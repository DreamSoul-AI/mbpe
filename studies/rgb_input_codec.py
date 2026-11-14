"""Reversible uint8 channel grids and capacity-matched CPU BF16 inputs.

Shape, transform and grammar are shared protocol state. Per-image token counts
are in-band; decoding an input tensor does not receive its unpadded byte length.
"""

from math import prod
import struct

import numpy as np

from bpe_block_file import pack_symbols, unpack_symbols


WIDTH = 192
TOKEN_BITS = 9


def validate_shape(shape):
    if (len(shape) != 3 or shape[0] not in (1, 3)
            or any(type(n) is not int or n <= 0 for n in shape)
            or shape[1] * shape[2] > 65535):
        raise ValueError("expected one or three nonempty uint16-sized channel grids")
    return tuple(shape)


def validate_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("uint8 channel-first image required")
    validate_shape(image.shape)


def difference(image):
    validate_image(image)
    result = image.copy()
    result[..., 1:] = np.subtract(image[..., 1:], image[..., :-1], dtype=np.uint8)
    return result


def restore_difference(image):
    validate_image(image)
    return np.cumsum(image, axis=-1, dtype=np.uint8)


def channel_samples(images):
    for image in images:
        validate_image(image)
        for channel in image:
            yield channel.shape, {point: int(value) for point, value in np.ndenumerate(channel)}


def validate_model(model):
    if model.rank != 2 or model.base != list(range(256)) or len(model.rules) > 256:
        raise ValueError("expected a rank-two byte alphabet and at most 256 rules")


def encode_bpe(image, model):
    validate_image(image)
    validate_model(model)
    counts, tokens = [], []
    for shape, cells in channel_samples([image]):
        packet = model.encode(shape, cells)
        if packet['literals'] or packet['domain'] is not None:
            raise ValueError("dense bytes must not require literals or a domain payload")
        counts.append(len(packet['tokens']))
        tokens.extend(packet['tokens'])
    return (struct.pack('<' + 'H' * len(counts), *counts)
            + pack_symbols(tokens, TOKEN_BITS, len(model.shapes))), counts


def payload_length(padded, shape, *, bpe):
    channels, height, width = validate_shape(shape)
    if not bpe:
        return prod(shape)
    header = channels * 2
    if len(padded) < header:
        raise ValueError("truncated channel counts")
    counts = struct.unpack('<' + 'H' * channels, padded[:header])
    if any(not 1 <= count <= height * width for count in counts):
        raise ValueError("channel token count outside the grid")
    return header + (sum(counts) * TOKEN_BITS + 7) // 8


def decode_bpe(blob, model, shape):
    validate_model(model)
    channels, height, width = validate_shape(shape)
    if len(blob) != payload_length(blob, shape, bpe=True):
        raise ValueError("incorrect BPE payload length")
    header = channels * 2
    counts = struct.unpack('<' + 'H' * channels, blob[:header])
    tokens = unpack_symbols(blob[header:], sum(counts), TOKEN_BITS, len(model.shapes))
    grids, start = [], 0
    for count in counts:
        packet = {'format': 'nd-bpe-candidate-packet', 'version': 1,
                  'shape': [height, width], 'domain': None, 'literals': [],
                  'tokens': tokens[start:start + count]}
        decoded_shape, cells = model.decode(packet)
        if decoded_shape != (height, width):
            raise ValueError("decoded shape differs")
        grids.append(np.fromiter((cells[p] for p in sorted(cells)), dtype=np.uint8).reshape(height, width))
        start += count
    return np.stack(grids)


def positions(byte_count, capacity):
    if type(capacity) is not int or not 1 <= capacity <= WIDTH or byte_count < 0:
        raise ValueError("invalid byte count or vector capacity")
    return (byte_count + capacity - 1) // capacity


def to_bf16(blob, capacity, torch):
    count = positions(len(blob), capacity)
    padded = blob + bytes(count * capacity - len(blob))
    values = torch.frombuffer(bytearray(padded), dtype=torch.uint8).reshape(count, capacity)
    tensor = torch.zeros((count, WIDTH), dtype=torch.bfloat16, device='cpu')
    tensor[:, :capacity] = values
    if tensor.numel() * tensor.element_size() != count * WIDTH * 2:
        raise ValueError("input tensor byte accounting differs")
    return tensor


def from_bf16(tensor, capacity, shape, *, bpe, torch):
    positions(0, capacity)
    if (tensor.device.type != 'cpu' or tensor.dtype != torch.bfloat16
            or tensor.ndim != 2 or tensor.shape[1] != WIDTH):
        raise ValueError("CPU BF16 vectors of width 192 required")
    values = tensor[:, :capacity]
    if (not torch.isfinite(values).all() or torch.any(values < 0) or torch.any(values > 255)
            or torch.any(values != values.to(torch.uint8).to(torch.bfloat16))
            or torch.any(tensor[:, capacity:] != 0)):
        raise ValueError("input violates byte-valued vector contract")
    padded = values.to(torch.uint8).reshape(-1).numpy().tobytes()
    length = payload_length(padded, shape, bpe=bpe)
    if tensor.shape[0] != positions(length, capacity) or any(padded[length:]):
        raise ValueError("noncanonical vector length, truncation or padding")
    return padded[:length]


def restore_input(tensor, capacity, shape, model, *, residual, bpe, torch):
    blob = from_bf16(tensor, capacity, shape, bpe=bpe, torch=torch)
    image = decode_bpe(blob, model, shape) if bpe else np.frombuffer(blob, dtype=np.uint8).reshape(shape)
    return restore_difference(image) if residual else image
