# Tokenization using Multidimensional Byte Pair Encoding (MBPE)
This repository implements a tokenization method using Multidimensional Byte Pair Encoding (MBPE). The input is intended to be any types of data, all of which are then converted into tuples for BPE tokenization and subsequent use in downstream tasks such as autoregressive modeling.


## Repository structure

- [Tuple-based implementation](mbpe/) contains the original tokenizer components.
- [Tensor implementation](src/mbpe/) contains patching, shuffling, tokenization and compression metrics.
- [MNIST example](src/main.py) exercises the tensor implementation. Running it can download MNIST and train a tokenizer.
- [Downstream training prototype](pipeline/) contains the separate model and training scripts.

## Getting started

Clone [DreamSoul-AI/mbpe](https://github.com/DreamSoul-AI/mbpe) and review the entry point for the implementation you need. The repository retains a [historical dependency list](requirements.txt). Check the required versions in an isolated environment before running an example.

## Repository maintenance

See the [repository conventions](REPO.md) and [engineering document index](docs/README.md) for maintenance.
