from collections import defaultdict
from .utils import dfs


class Vocab:
    def __init__(self):
        self.symbol2codeword = {}  # maps symbol -> codeword
        self.codeword2symbol = {}  # maps codeword -> symbol
        self.codeword_size_symbols = set()
        self.codeword_size_codewords = set()

    def add(self, symbol, codeword):
        if symbol in self.symbol2codeword or codeword in self.codeword2symbol:
            raise ValueError("Symbol or codeword already exists.")
        self.symbol2codeword[symbol] = codeword
        self.codeword2symbol[codeword] = symbol
        self.codeword_size_symbols.add(symbol)
        self.codeword_size_codewords.add(codeword)
        return

    def update(self, symbols):
        start_index = len(self.symbol2codeword)
        new_entries = 0
        codewords = []

        if isinstance(symbols, list):
            for symbol in symbols:
                if symbol not in self.symbol2codeword:
                    codeword = str(start_index + new_entries)
                    self.add(symbol, codeword)
                    new_entries += 1
                else:
                    codeword = self.symbol2codeword[symbol]
                codewords.append(codeword)
        else:
            symbol = symbols
            if symbol not in self.symbol2codeword:
                codeword = str(start_index + new_entries)
                self.add(symbol, codeword)
                new_entries += 1
            else:
                codeword = self.symbol2codeword[symbol]
            codewords = codeword
        return codewords

    def filter_symbols(self, symbols, codeword_size=None):
        in_symbols = []
        in_indices = []
        out_symbols = []
        out_indices = []

        for idx, symbol in enumerate(symbols):
            is_in = False
            if symbol in self.symbol2codeword:
                if codeword_size is not None:
                    if symbol in self.codeword_size_symbols:
                        is_in = True
                else:
                    is_in = True
            if is_in:
                in_symbols.append(symbol)
                in_indices.append(idx)
            else:
                out_symbols.append(symbol)
                out_indices.append(idx)

        return in_symbols, in_indices, out_symbols, out_indices

    def get_codeword(self, symbol, codeword_size=None, default=None):
        if codeword_size is not None:
            if symbol in self.codeword_size_symbols:
                codeword = self.symbol2codeword.get(symbol, default)
            else:
                codeword = default
        else:
            codeword = self.symbol2codeword.get(symbol, default)
        return codeword

    def get_codewords(self, symbols, codeword_size=None, default=None):
        codewords = []
        for symbol in symbols:
            codeword = self.get_codeword(symbol, codeword_size, default)
            if codeword is not None:
                codewords.append(codeword)
        return codewords

    def get_symbol(self, codeword, codeword_size=None, default=None):
        if codeword_size is not None:
            if codeword in self.codeword_size_codewords:
                symbol = self.codeword2symbol.get(codeword, default)
            else:
                symbol = default
        else:
            symbol = self.codeword2symbol.get(codeword, default)
        return symbol

    def get_symbols(self, codewords, codeword_size=None, default=None):
        symbols = []
        for codeword in codewords:
            symbol = self.get_symbol(codeword, codeword_size, default)
            if symbol is not None:
                symbols.append(symbol)
        return symbols

    def __contains__(self, key, codeword_size=None):
        return key in self.symbol2codeword or key in self.codeword2symbol

    def __len__(self):
        return len(self.symbol2codeword)

    def __repr__(self):
        num_entries = len(self)
        preview = list(self.symbol2codeword.items())
        preview_str = ", ".join(f"{repr(k)}: {repr(v)}" for k, v in preview)
        return f"Vocab({num_entries} entries: {{{preview_str}}})"
