"""Finite-window sorted batches with explicit CPU BF16 input accounting."""

import struct


def window_groups(lengths, arrival, window, batch_size=16, *, sort=True):
    if (type(window) is not int or type(batch_size) is not int or batch_size <= 0
            or window < batch_size or window % batch_size):
        raise ValueError('window must be a positive multiple of batch size')
    if any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError('positive integer lengths required')
    if sorted(arrival) != list(range(len(lengths))):
        raise ValueError('arrival must be a permutation of sample identities')
    for start in range(0, len(arrival), window):
        queue = list(arrival[start:start+window])
        order = sorted(queue, key=lambda i: lengths[i]) if sort else queue
        yield queue, [order[i:i+batch_size] for i in range(0,len(order),batch_size)]


def make_batch(tensors, identities, torch):
    if not identities or len(set(identities)) != len(identities) or any(i < 0 for i in identities):
        raise ValueError('nonempty distinct nonnegative identities required')
    if any(t.device.type != 'cpu' or t.dtype != torch.bfloat16 or t.ndim != 2
           or t.shape[1] != 192 or not len(t) for t in tensors):
        raise ValueError('nonempty CPU BF16 [positions,192] inputs required')
    if len(tensors) != len(identities):
        raise ValueError('one identity per tensor required')
    lengths = torch.tensor([len(t) for t in tensors], dtype=torch.int32)
    batch = torch.zeros((len(tensors), int(lengths.max()), 192), dtype=torch.bfloat16)
    mask = torch.zeros(batch.shape[:2], dtype=torch.bool)
    for row, tensor in enumerate(tensors):
        batch[row,:len(tensor)] = tensor
        mask[row,:len(tensor)] = True
    return batch, lengths, mask, tuple(identities)


def batch_bytes(batch, lengths, mask):
    return {'input_bytes':batch.numel()*batch.element_size(),
            'length_bytes':lengths.numel()*lengths.element_size(),
            'mask_bytes':mask.numel()*mask.element_size(),
            'allocated_positions':batch.shape[0]*batch.shape[1]}


def unpack_batch(batch, lengths, mask, identities, torch):
    if (batch.device.type != 'cpu' or batch.dtype != torch.bfloat16 or batch.ndim != 3
            or batch.shape[2] != 192 or lengths.device.type != 'cpu' or lengths.dtype != torch.int32
            or lengths.shape != (len(batch),) or mask.device.type != 'cpu' or mask.dtype != torch.bool
            or tuple(mask.shape) != tuple(batch.shape[:2]) or len(identities) != len(batch)
            or len(set(identities)) != len(identities)):
        raise ValueError('invalid batch metadata')
    result = []
    for row, identity in enumerate(identities):
        length = int(lengths[row])
        if identity < 0 or not 1 <= length <= batch.shape[1]:
            raise ValueError('invalid identity or length')
        if (not bool(mask[row,:length].all()) or bool(mask[row,length:].any())
                or bool((batch[row,length:] != 0).any())):
            raise ValueError('invalid prefix mask or nonzero padding')
        result.append((identity,batch[row,:length].clone()))
    return result


def write_batch(handle, packet):
    batch,lengths,mask,identities = packet
    handle.write(struct.pack('<II',batch.shape[0],batch.shape[1]))
    handle.write(struct.pack('<'+'I'*len(identities),*identities))
    handle.write(lengths.numpy().astype('<i4').tobytes())
    handle.write(mask.numpy().tobytes())
    # Reinterpret BF16 bits without numerical conversion.
    import torch
    handle.write(batch.view(torch.uint16).numpy().astype('<u2').tobytes())


def read_batch(handle, torch):
    prefix = handle.read(8)
    if not prefix:
        return None
    if len(prefix) != 8:
        raise ValueError('truncated batch header')
    count,maximum = struct.unpack('<II',prefix)
    if not 1 <= count <= 16 or not 1 <= maximum <= 19:
        raise ValueError('frame outside fixed RGB protocol')
    def exact(n):
        data = handle.read(n)
        if len(data) != n:
            raise ValueError('truncated batch')
        return bytearray(data)
    identities = struct.unpack('<'+'I'*count,exact(4*count))
    lengths = torch.frombuffer(exact(4*count),dtype=torch.int32).clone()
    raw_mask = exact(count*maximum)
    if any(v not in (0,1) for v in raw_mask):
        raise ValueError('mask must contain canonical boolean bytes')
    mask = torch.frombuffer(raw_mask,dtype=torch.bool).reshape(count,maximum).clone()
    batch = torch.frombuffer(exact(count*maximum*384),dtype=torch.bfloat16).reshape(count,maximum,192).clone()
    unpack_batch(batch,lengths,mask,identities,torch)
    return batch,lengths,mask,identities
