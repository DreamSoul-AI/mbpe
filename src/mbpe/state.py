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
        total_length = len(symbols) + len(codewords)
        merged = np.empty(total_length, dtype=object)
        merged[symbol_indices] = np.fromiter(symbols, dtype=object)
        merged[codeword_indices] = codewords
        joined = merged.tolist()
        return joined

    def merge(self, symbols):
        merged = []
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            merged.append(pair)
        return merged

    def merge_symbol(self, symbols, symbol, codeword):
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

    def __repr__(self):
        return (f"State(\n"
                f"  data=dtype({self.dtype}), size={self.size},\n"
                f"  symbols={self.symbols},\n"
                f"  symbol_indices={self.symbol_indices},\n"
                f"  codewords={self.codewords},\n"
                f"  codeword_indices={self.codeword_indices},\n"
                f"  joined={self.joined}\n"
                f")")
