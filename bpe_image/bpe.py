import numpy as np
from torchvision import datasets, transforms
from transformers import BertTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from collections import Counter, defaultdict

"""
Final vocab should be something like
(0, 255, 3, ... ): # number of times

Run Bpe on a list [[0, 255, 3, 44], [0, 23, 24], [3, 23, 21, etc]] 
Reshape numpy first
then make it into a list
make it into a list of strings
then BPE it
"""

def bpe(strings, num_merges):
    # prepare the initial data as a list of lists where each inner list is a sequence of elements (words)
    data = [s.split() for s in strings]

    # initialize the sequence counts
    sequence_counts = Counter()
    for sequence in data:
        for i in range(len(sequence)):
            for j in range(i + 1, len(sequence) + 1):
                sequence_counts[tuple(sequence[i:j])] += 1

    for _ in range(num_merges):
        # find the most common sequence
        most_common_sequence = max(sequence_counts, key=sequence_counts.get, default=None)
        if not most_common_sequence or len(most_common_sequence) <= 1:
            break

        # create merged token for most common sequence
        merged_token = ' '.join(most_common_sequence)
        new_sequence_counts = Counter()

        # update data and counter with merged token
        for sequence in data:
            new_sequence = []
            skip = 0
            for i in range(len(sequence)):
                if skip:
                    skip -= 1
                    continue
                if tuple(sequence[i:i+len(most_common_sequence)]) == most_common_sequence:
                    new_sequence.append(merged_token)
                    skip = len(most_common_sequence) - 1
                else:
                    new_sequence.append(sequence[i])
            for i in range(len(new_sequence)):
                for j in range(i + 1, len(new_sequence) + 1):
                    new_sequence_counts[tuple(new_sequence[i:j])] += 1
            data = [new_sequence]

        sequence_counts = new_sequence_counts

    return sequence_counts



# load mnist
mnist_train = datasets.MNIST(root="./data", train=True, download=True)
subset_size = 1
subset_images = mnist_train.data[:subset_size].numpy()

# reshape subset to (100, 28*28) and convert to binary strings
reshaped_subset = subset_images.reshape(subset_size, -1) # 2d array each row = image
string_lists = [[str(element) for element in row] for row in reshaped_subset]

num_merges = 10
one_image = " ".join(string_lists[0])
example_test = ["0 0 0 1 23 233 232"]
vocab = bpe(example_test, num_merges)
print(vocab)