"""
Modified from Minimal (byte-level) Byte Pair Encoding tokenizer.
Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, threshold=2, resume=True, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        ids = text
        # iteratively merge the most common pairs to create new tokens
        if resume:
            merges = self.merges  # (int, int) -> int
            vocab = {idx: idx for idx in range(256)}
            for (p0, p1), idx in self.merges.items():
                vocab[idx] = (p0, p1)
        elif not resume:
            merges = {}  # (int, int) -> int
            vocab = {idx: idx for idx in range(256)}
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # If the number of occurrences is below the threshold, it will no longer be merged.
            occur = max(stats.values())
            if occur < threshold:
                break
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = pair
            # prints
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()

    def decode(self, ids):
        num_merges = len(self.vocab) - 256
        print("--------------------\n", "len(self.vocab): ", len(self.vocab), "\n--------------------")
        temp = []
        while num_merges > 0:
            for idx in ids:
                if idx < 256:
                    temp.append(self.vocab[idx])
                else:
                    (p0, p1) = self.vocab[idx]
                    temp.append(p0)
                    temp.append(p1)

            num_merges -= 1
            ids = temp
            temp = []
        text = ids
        return text

    def encode(self, text):
        ids = text
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
