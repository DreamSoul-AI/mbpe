"""MBPE research package. Heavy numerical modules load on first access."""

from importlib import import_module

__all__ = ["tokenizer", "utils", "tensor_shuffle"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f".structure.bpe.{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
