"""Small, explicit configuration for the existing tensor tokenizer."""

from dataclasses import asdict, dataclass
import math


DEFAULT_EXPERIMENT_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class TokenizerConfig:
    max_codeword_size: tuple[int, ...] = (1, 2, 2)
    batch_size: int = 4
    root_min_freq: float = 0.01
    merge_min_freq: float = 0.001
    counter_min_freq: float = 0.01
    seed: int = 0
    max_symbols: int | None = None

    def __post_init__(self):
        if not self.max_codeword_size or any(type(x) is not int or x <= 0 for x in self.max_codeword_size):
            raise ValueError("max_codeword_size must contain positive integers")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        for value in (self.root_min_freq, self.merge_min_freq, self.counter_min_freq):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("frequency thresholds must be finite numbers in [0, 1]")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.max_symbols is not None and (type(self.max_symbols) is not int or self.max_symbols <= 0):
            raise ValueError("max_symbols must be a positive integer or omitted")

    def to_dict(self):
        return asdict(self)

    def tokenizer_kwargs(self):
        return {
            "max_codeword_size": self.max_codeword_size,
            "batch_size": self.batch_size,
            "max_workers": 0,
            "min_freq": {"root": self.root_min_freq, "merge": self.merge_min_freq,
                         "freq_counter": self.counter_min_freq},
            "seed": self.seed,
            "max_symbols": self.max_symbols,
        }
