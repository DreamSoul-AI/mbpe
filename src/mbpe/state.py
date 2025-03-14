class State:
    def __init__(self, data, code_size, tuple_list=None, tuple_indices=None, code_list=None, code_indices=None,
                 joined_list=None):
        self.data = data
        self.dtype = data.dtype
        self.size = list(data.size())
        self.code_size = code_size

        self.tuple_list = tuple_list if tuple_list is not None else []
        self.tuple_indices = tuple_indices if tuple_indices is not None else []
        self.code_list = code_list if code_list is not None else []
        self.code_indices = code_indices if code_indices is not None else []
        self.joined_list = joined_list if joined_list is not None else []

    def __repr__(self):
        return (f"State(\n"
                f"  data=dtype({self.dtype}), size={self.size},\n"
                f"  code_size={self.code_size},\n"
                f"  tuple_indices={self.tuple_indices},\n"
                f"  code_list={self.code_list},\n"
                f"  code_indices={self.code_indices},\n"
                f"  joined_list={self.joined_list}\n"
                f")")