import torch
import torch.nn as nn
import math


class Patchify(nn.Module):
    def __init__(self, patch_size):
        super(Patchify, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        if x.dim() == len(self.patch_size):
            raise ValueError('Need a sequence dimension')
        dim_index = list(range(x.dim() - len(self.patch_size), x.dim()))
        downscale_factor = [x.shape[dim_index[i]] // self.patch_size[i] for i in range(len(dim_index))]
        x = self._tensor_unshuffle(x, downscale_factor, dim_index)
        return x

    @staticmethod
    def _tensor_unshuffle(tensor, downscale_factor, dim_index):
        tensor_size = list(tensor.size())
        pre_index = list(range(tensor.dim() - len(dim_index)))
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
        permute_order = pre_index + downscale_factor_index + base_index
        permuted_tensor = reshaped_tensor.permute(permute_order)

        pre_size = [tensor_size[i] for i in pre_index[:len(pre_index) - 1]]
        downscale_size = [tensor_size[i] for i in downscale_factor_index]
        merged_size = math.prod(downscale_size) * tensor_size[dim_index[0] - 1]
        base_size = [tensor_size[i] for i in base_index]
        unshuffled_size = pre_size + [merged_size] + base_size
        unshuffled_tensor = permuted_tensor.reshape(unshuffled_size)
        return unshuffled_tensor


class Reconstruct(nn.Module):
    def __init__(self, patch_size):
        super(Reconstruct, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        dim_index = list(range(x.dim() - len(self.patch_size), x.dim()))
        upscale_factor = [self.patch_size[i] // x.shape[dim_index[i]] for i in range(len(dim_index))]
        x = self._tensor_shuffle(x, upscale_factor, dim_index)
        return x

    @staticmethod
    def _tensor_shuffle(tensor, upscale_factor, dim_index):
        tensor_size = list(tensor.size())
        print(tensor_size)
        pre_index = list(range(tensor.dim() - len(dim_index)))
        pre_size = [tensor_size[i] for i in pre_index[:len(pre_index) - 1]] + [1]
        base_size = [tensor_size[i] for i in dim_index]
        unmerged_size = pre_size + upscale_factor + base_size
        unmerged_tensor = tensor.view(unmerged_size)

        pre_index = list(range(len(pre_size)))
        permute_order = pre_index
        base_index = list(range(len(upscale_factor) + len(pre_index),
                                len(upscale_factor) + len(pre_index) + len(base_size)))
        permute_order = permute_order + base_index
        downscale_factor_index = list(range(len(pre_index), len(dim_index) + len(pre_index)))
        index_accum = 0
        for i in range(len(dim_index)):
            dim_index_i = dim_index[i] + index_accum
            permute_order.insert(dim_index_i + 1, downscale_factor_index[i])
            index_accum += 1
        permuted_tensor = unmerged_tensor.permute(permute_order)
        shuffled_tensor_size = pre_size + [base_size[i] * upscale_factor[i] for i in range(len(dim_index))]
        shuffled_tensor = permuted_tensor.reshape(shuffled_tensor_size)
        return shuffled_tensor
