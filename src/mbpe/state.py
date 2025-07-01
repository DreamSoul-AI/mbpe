import numpy as np


class State:
    def __init__(self, data, symbols=None, symbol_indices=None, codewords=None, codeword_indices=None,
                 joined=None):
        self.data = data
        self.dtype = data.dtype
        self.size = list(data.size())

        self.symbols = symbols if symbols is not None else []
        self.symbol_indices = symbol_indices if symbol_indices is not None else []
        self.codewords = codewords if codewords is not None else []
        self.codeword_indices = codeword_indices if codeword_indices is not None else []
        self.joined = joined if joined is not None else []

    def update(self, data=None, symbols=None, symbol_indices=None, codewords=None, codeword_indices=None, joined=None):
        if data is not None:
            self.data = data
            self.dtype = data.dtype
            self.size = list(data.size())

        if symbols is not None:
            self.symbols = symbols

        if symbol_indices is not None:
            self.symbol_indices = symbol_indices

        if codewords is not None:
            self.codewords = codewords

        if codeword_indices is not None:
            self.codeword_indices = codeword_indices

        if joined is not None:
            self.joined = joined
        return

    def join(self, symbols, symbol_indices, codewords, codeword_indices):
        print(symbols)
        print(codewords)
        total_length = len(symbols) + len(codewords)
        merged = np.empty(total_length, dtype=object)
        merged[symbol_indices] = np.fromiter(symbols, dtype=object)
        merged[codeword_indices] = codewords
        joined = merged.tolist()
        print(joined)
        return joined

    def split(self, joined, scale_factor):
        symbols = []
        symbol_indices = []
        codewords = []
        codeword_indices = []

        for item in joined:
            base_offset = len(symbols) * scale_factor + len(codewords)
            if isinstance(item, tuple):
                symbols.append(item)
                symbol_indices.extend(range(base_offset, base_offset + scale_factor))
            else:
                codewords.append(item)
                codeword_indices.append(base_offset)
        return symbols, symbol_indices, codewords, codeword_indices

    def pair(self, input):
        pairs = []
        for i in range(len(input) - 1):
            pair = (input[i], input[i + 1])
            pairs.append(pair)
        return pairs

    def merge(self, symbols, symbol, codeword):
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == symbol:
                new_symbols.append(codeword)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        return new_symbols

    def finalize(self):
        self.symbols = []
        self.symbol_indices = []
        self.codewords = []
        self.codeword_indices = []
        return

    def __repr__(self):
        return (
            f"State(\n"
            f"  data=Tensor(dtype={self.dtype}, size={self.size}),\n"
            f"  symbols={repr(self.symbols)},\n"
            f"  symbol_indices={repr(self.symbol_indices)},\n"
            f"  codewords={repr(self.codewords)},\n"
            f"  codeword_indices={repr(self.codeword_indices)},\n"
            f"  joined={repr(self.joined)}\n"
            f")"
        )
