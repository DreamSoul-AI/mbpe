import math

def tensor_unshuffle(tensor, downscale_factor, dim_index):
    tensor_size = list(tensor.size())
    base_index = []
    downscale_factor_index = []
    index_accum = 0
    for i in range(len(dim_index)):
        dim_index_i = dim_index[i] + index_accum
        tensor_size[dim_index_i] //= downscale_factor[i]
        tensor_size.insert(dim_index_i + 1, downscale_factor[i])
        base_index.append(dim_index_i)
        downscale_factor_index.append(dim_index_i + 1)
        index_accum += 1
    reshaped_tensor = tensor.view(tensor_size)
    permute_order = [0] + downscale_factor_index + base_index
    permuted_tensor = reshaped_tensor.permute(permute_order)
    downscale_size = [tensor_size[i] for i in downscale_factor_index]
    base_size = [tensor_size[i] for i in base_index]
    merged_size = math.prod(downscale_size)
    unshuffled_size = [tensor_size[0]] + [merged_size] + base_size
    unshuffled_tensor = permuted_tensor.reshape(unshuffled_size)
    return unshuffled_tensor


def tensor_shuffle(tensor, upscale_factor, dim_index):
    tensor_size = list(tensor.size())
    base_size = list(tensor_size[2:])
    unmerged_size = [tensor_size[0]] + upscale_factor + base_size
    unmerged_tensor = tensor.view(unmerged_size)
    base_index = list(range(len(upscale_factor) + 1, len(upscale_factor) + 1 + len(base_size)))
    downscale_factor_index = list(range(1, len(dim_index) + 1))
    permute_order = [0] + base_index
    index_accum = 0
    for i in range(len(dim_index)):
        dim_index_i = dim_index[i] + index_accum
        permute_order.insert(dim_index_i + 1, downscale_factor_index[i])
        index_accum += 1
    permuted_tensor = unmerged_tensor.permute(permute_order)
    shuffle_tensor_size = [tensor_size[0]] + [base_size[i] * upscale_factor[i] for i in range(len(dim_index))]
    shuffle_tensor = permuted_tensor.reshape(shuffle_tensor_size)
    return shuffle_tensor
