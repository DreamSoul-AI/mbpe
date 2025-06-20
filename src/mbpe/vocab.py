class Vocab:
    def __init__(self):
        self.symbol2codeword = {}  # maps symbol -> codeword
        self.codeword2symbol = {}  # maps codeword -> symbol

    def add(self, symbol, codeword):
        if symbol in self.symbol2codeword or codeword in self.codeword2symbol:
            raise ValueError("Symbol or codeword already exists.")
        self.symbol2codeword[symbol] = codeword
        self.codeword2symbol[codeword] = symbol
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

    def filter_symbols(self, symbols):
        in_symbols = []
        in_indices = []
        out_symbols = []
        out_indices = []

        for idx, symbol in enumerate(symbols):
            if symbol in self.symbol2codeword:
                in_symbols.append(symbol)
                in_indices.append(idx)
            else:
                out_symbols.append(symbol)
                out_indices.append(idx)

        return in_symbols, in_indices, out_symbols, out_indices

    def get_codeword(self, symbol, default=None):
        return self.symbol2codeword.get(symbol, default)

    def get_codewords(self, symbols):
        codewords = []
        for symbol in symbols:
            codeword = self.symbol2codeword.get(symbol)
            if codeword is not None:
                codewords.append(codeword)
        return codewords

    def get_symbol(self, codeword, default=None):
        return self.codeword2symbol.get(codeword, default)

    def __contains__(self, key):
        return key in self.symbol2codeword or key in self.codeword2symbol

    def __len__(self):
        return len(self.symbol2codeword)

    def __repr__(self):
        num_entries = len(self)
        preview = list(self.symbol2codeword.items())  # Show all entries
        preview_str = ", ".join(f"{repr(k)}: {repr(v)}" for k, v in preview)
        return f"Vocab({num_entries} entries: {{{preview_str}}})"