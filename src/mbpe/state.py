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

    def __repr__(self):
        return (f"State(\n"
                f"  data=dtype({self.dtype}), size={self.size},\n"
                f"  tuple_list={self.symbols},\n"
                f"  tuple_indices={self.symbol_indices},\n"
                f"  code_list={self.codewords},\n"
                f"  code_indices={self.codeword_indices},\n"
                f"  joined_list={self.joined}\n"
                f")")
