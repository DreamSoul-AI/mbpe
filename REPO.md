# MBPE repository conventions

This checkout contains research implementations of multidimensional byte pair encoding. Use the [engineering document index](docs/README.md) to locate the relevant implementation and its examples.

## Implementation and validation

- The root [mbpe/](mbpe/) and [src/mbpe/](src/mbpe/) contain different implementations. Confirm which package an entry point resolves before comparing behavior.
- [src/main.py](src/main.py) enables MNIST download and tokenizer training. Review those effects before choosing a validation command. Keep checks on synthetic inputs separate from full-data experiments.
- [src/test.py](src/test.py) is a patching example. Its presence is evidence of a reference script. Record the checks that actually ran, including the import path, dependencies and inputs.
- [pipeline/](pipeline/) is the downstream training prototype. Confirm model, data and compute scope before running its entry points.

Keep development, research and run records with their actual source version and evidence. Use the project's current instructions when introducing new research documents or execution paths.
