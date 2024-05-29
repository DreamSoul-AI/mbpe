# Tokenization using Multidimensional Byte Pair Encoding (MBPE)
This repository implements a tokenization method using Multidimensional Byte Pair Encoding (MBPE). The input is intended to be any types of data, all of which are then converted into tuples for BPE tokenization and subsequent use in downstream tasks such as autoregressive modeling.

## File Structure
The repository has the following structure:
- [mbpe/base.py](./mbpe/base.py): The Tokenizer class is implemented as the fundamental class. It contains the train_encode and decode placeholders, along with save/load features. It is intended for inheritance rather than direct utilization.
- [mbpe/basic.py](./mbpe/basic.py): Implements the basic tokenizer.
- [mbpe/utils.py](./mbpe/utils.py): Implements several utility functions including reshaping, merge, etc.
- [examples/main.py](./examples/main.py): The main Python script that performs tokenization using MDBPE on images from the MNIST dataset.

## Getting Started
To use this script, follow these steps:
1. Clone the repository to your local machine:
```
    git clone https://github.com/Collaborative-AI/multimodality.git
```
2. Navigate to the cloned directory
3. Install Anaconda and create a new Anaconda virtual environment
4. Activate the virtual environment
5. Install the required dependencies:
```
    pip install -r requirements.txt
```
6. Run the main script:
```
    python .\examples\main.py
```