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

    def get_codeword(self, symbol, default=None):
        return self.symbol2codeword.get(symbol, default)

    def get_symbol(self, codeword, default=None):
        return self.codeword2symbol.get(codeword, default)

    def __contains__(self, key):
        return key in self.symbol2codeword or key in self.codeword2symbol

    def __len__(self):
        return len(self.symbol2codeword)

    def __repr__(self):
        num_entries = len(self)
        preview_limit = 20
        preview = list(self.symbol2codeword.items())[:preview_limit]
        preview_str = ", ".join(f"{repr(k)}: {repr(v)}" for k, v in preview)
        if num_entries > preview_limit:
            preview_str += ", ..."
        return f"Vocab({num_entries} entries: {{{preview_str}}})"
