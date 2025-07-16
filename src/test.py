import torch
from mbpe.patch import Patchify, Reconstruct


def main():
    # def dfs(token, vocab, codeword_size, codeword_size_codewords):
    #     """
    #     Recursively decode a codeword only if it's registered under the current codeword_size.
    #
    #     Args:
    #         token (str): The codeword to decode.
    #         vocab (dict): Dictionary mapping strings to tuples of either strings or ints.
    #         codeword_size (tuple): Current decoding level.
    #         codeword_size_codewords (dict[tuple, set[str]]): Valid codewords per codeword size.
    #
    #     Returns:
    #         Union[tuple[int], str]: Fully expanded tuple if token is decodable at this level, else str token.
    #     """
    #     if token not in codeword_size_codewords.get(codeword_size, set()):
    #         return token  # Defer decoding — not created at this level
    #
    #     result = []
    #     stack = [token]
    #
    #     while stack:
    #         current = stack.pop()
    #         if isinstance(current, str):
    #             if current not in codeword_size_codewords.get(codeword_size, set()):
    #                 result.append(current)
    #             else:
    #                 resolved = vocab[current]
    #                 stack.extend(reversed(resolved))
    #         elif isinstance(current, int):
    #             result.append(current)
    #         else:
    #             raise TypeError(f"Unsupported element in vocab: {current}")
    #
    #     return tuple(reversed(result)) if all(isinstance(x, int) for x in result) else tuple(result)
    #
    # vocab = {
    #     'A': (1, 2),
    #     'B': ('A', 'A'),  # expands to (1, 2, 1, 2)
    #     'C': ('B', 'B'),  # expands to (1, 2, 1, 2, 1, 2, 1, 2)
    #     'X': ('C',),  # but suppose 'X' was created at a higher level
    # }
    #
    # codeword_size_codewords = {
    #     (1, 2): {'A', 'B', 'C'},
    #     (2, 2): {'X'}  # 'X' must not be decoded at level (1, 2)
    # }
    #
    # print(dfs('C', vocab, (1, 2), codeword_size_codewords))  # (1, 2, 1, 2, 1, 2, 1, 2)
    # print(dfs('X', vocab, (1, 2), codeword_size_codewords))  # 'X' (left as-is)

    # def dfs(input, search_fn):
    #     if isinstance(input, int):
    #         return [input]
    #     elif isinstance(input, str):
    #         return dfs(search_fn(input), search_fn)
    #     elif isinstance(input, tuple):
    #         result = []
    #         for item in input:
    #             result.extend(dfs(item, search_fn))
    #         return result
    #     else:
    #         raise TypeError(f"Unexpected type {type(input)} in dfs")
    #
    # # Sample vocabulary: string -> tuple of int or other strings
    # vocab = {
    #     'A': ('B', 'C'),
    #     'B': (1, 2),
    #     'C': ('D', 3),
    #     'D': (4, 5),
    #     'X': (6,)  # Also allow direct mapping to int
    # }
    #
    # # search_fn using the vocab dictionary
    # def search_fn(key):
    #     if key not in vocab:
    #         raise KeyError(f"Key '{key}' not found in vocabulary.")
    #     return vocab[key]
    #
    # # Test cases
    # test_cases = [
    #     ('A', [1, 2, 4, 5, 3]),
    #     (('A', 7), [1, 2, 4, 5, 3, 7]),
    #     ((('B', 'C'), 'D'), [1, 2, 4, 5, 3, 4, 5]),
    #     (9, [9]),
    #     ('X', [6])
    # ]
    #
    # # Run the test cases
    # for i, (input_data, expected) in enumerate(test_cases, 1):
    #     try:
    #         result = dfs(input_data, search_fn)
    #         assert result == expected, f"Expected {expected}, got {result}"
    #         print(f"Test case {i} passed: {input_data} → {result}")
    #     except Exception as e:
    #         print(f"Test case {i} failed: {input_data} → Error: {e}")

    # Define sample input tensor
    batch_size = 2
    channels = 3
    height = 8
    width = 8
    is_batch = True
    patch_size = [1, 2, 2]
    dim_index = [2, 3, 4]  # Apply patching on H and W

    x = torch.arange(batch_size * channels * height * width).float().reshape(batch_size, channels, height, width)
    x = x.unsqueeze(1)

    if not is_batch:
        x = x[0]
        dim_index = [1, 2, 3]

    print("Original Input Shape:", x.shape)

    # Patchify setup: divide height and width by 2

    patchify = Patchify(patch_size=patch_size, dim_index=dim_index)
    x_patched = patchify(x, is_batch=is_batch)

    print("Patched Output Shape:", x_patched.shape)

    # Reconstruct setup: needs original data size
    data_size = list(x.shape)
    reconstruct = Reconstruct(data_size=data_size, dim_index=dim_index)
    x_reconstructed = reconstruct(x_patched, is_batch=is_batch)

    print("Reconstructed Output Shape:", x_reconstructed.shape)

    # Check reconstruction correctness
    if torch.allclose(x, x_reconstructed):
        print("✅ Reconstruction matches original input.")
    else:
        print("❌ Reconstruction does not match original input.")
    return


if __name__ == "__main__":
    main()
