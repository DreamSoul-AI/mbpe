import torch
import torch.nn as nn
import math


class Patchify(nn.Module):
    def __init__(self, patch_size, dim_index):
        """
        Initializes the Patchify module.
        :param target_patch_shape: Target shape for patches, e.g., (1, 7, 7).
        """
        super(Patchify, self).__init__()
        self.patch_size = patch_size
        self.dim_index = dim_index
        print(dim_index, patch_size)

    def get_scale_factor(self, size):
        """
        Computes the scale factor for downscaling based on the patch size.
        :param size: Original size of the tensor.
        :return: Scale factor.
        """
        for i, dim in enumerate(self.dim_index):
            factor = size[dim] // self.patch_size[i]
            if factor != 1:
                return factor
        return 1

    def forward(self, x, is_batch=True):
        """
        Patchify the input tensor.
        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: Patchified tensor.
        """
        if not is_batch:
            x = x.unsqueeze(0)
            dim_index = [idx + 1 for idx in self.dim_index]
        else:
            dim_index = self.dim_index
        # Dynamically calculate downscale factor
        downscale_factor = [
            x.shape[dim_index[i]] // self.patch_size[i] for i in range(len(dim_index))
        ]
        x = self._tensor_unshuffle(x, downscale_factor, dim_index)
        if not is_batch:
            x = x.squeeze(0)
        return x

    @staticmethod
    def _tensor_unshuffle(tensor, downscale_factor, dim_index):
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
        permute_order = [0, 1] + downscale_factor_index + base_index
        permuted_tensor = reshaped_tensor.permute(permute_order)
        downscale_size = [tensor_size[i] for i in downscale_factor_index]
        base_size = [tensor_size[i] for i in base_index]
        merged_size = math.prod(downscale_size) * tensor_size[1]
        unshuffled_size = [tensor_size[0]] + [merged_size] + base_size
        unshuffled_tensor = permuted_tensor.reshape(unshuffled_size)
        return unshuffled_tensor


class Reconstruct(nn.Module):
    def __init__(self, data_size, dim_index):
        """
        Initializes the Reconstruct module.
        :param original_shape: Original shape of the tensor to reconstruct, e.g., (batch_size, channels, height, width).
        """
        super(Reconstruct, self).__init__()
        self.data_size = data_size
        self.dim_index = dim_index

    def forward(self, x, is_batch=True):
        """
        Reconstruct the original tensor from patches.
        :param x: Patchified tensor.
        :return: Reconstructed tensor.
        """
        if not is_batch:
            x = x.unsqueeze(0)
            dim_index = [idx + 1 for idx in self.dim_index]
        else:
            dim_index = self.dim_index
        upscale_factor = [
            self.data_size[dim_index[i]] // x.shape[-len(dim_index):][i] for i in range(len(dim_index))
        ]
        x = self._tensor_shuffle(x, upscale_factor, dim_index)
        if not is_batch:
            x = x.squeeze(0)
        return x

    @staticmethod
    def _tensor_shuffle(tensor, upscale_factor, dim_index):
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
