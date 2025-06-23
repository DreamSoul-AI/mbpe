def main():
    def dfs(token, vocab, codeword_size, codeword_size_codewords):
        """
        Recursively decode a codeword only if it's registered under the current codeword_size.

        Args:
            token (str): The codeword to decode.
            vocab (dict): Dictionary mapping strings to tuples of either strings or ints.
            codeword_size (tuple): Current decoding level.
            codeword_size_codewords (dict[tuple, set[str]]): Valid codewords per codeword size.

        Returns:
            Union[tuple[int], str]: Fully expanded tuple if token is decodable at this level, else str token.
        """
        if token not in codeword_size_codewords.get(codeword_size, set()):
            return token  # Defer decoding — not created at this level

        result = []
        stack = [token]

        while stack:
            current = stack.pop()
            if isinstance(current, str):
                if current not in codeword_size_codewords.get(codeword_size, set()):
                    result.append(current)
                else:
                    resolved = vocab[current]
                    stack.extend(reversed(resolved))
            elif isinstance(current, int):
                result.append(current)
            else:
                raise TypeError(f"Unsupported element in vocab: {current}")

        return tuple(reversed(result)) if all(isinstance(x, int) for x in result) else tuple(result)

    vocab = {
        'A': (1, 2),
        'B': ('A', 'A'),  # expands to (1, 2, 1, 2)
        'C': ('B', 'B'),  # expands to (1, 2, 1, 2, 1, 2, 1, 2)
        'X': ('C',),  # but suppose 'X' was created at a higher level
    }

    codeword_size_codewords = {
        (1, 2): {'A', 'B', 'C'},
        (2, 2): {'X'}  # 'X' must not be decoded at level (1, 2)
    }

    print(dfs('C', vocab, (1, 2), codeword_size_codewords))  # (1, 2, 1, 2, 1, 2, 1, 2)
    print(dfs('X', vocab, (1, 2), codeword_size_codewords))  # 'X' (left as-is)


if __name__ == "__main__":
    main()
