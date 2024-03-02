from tensorflow.keras.datasets import mnist
import numpy as np
from collections import defaultdict, Counter

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def image_to_list_of_lists(image):
    return [[str(pixel) for pixel in row] for row in image]


"""
If i have image with pixels:

0, 0, 2, 2
1, 1, 0, 0,
2, 2, 0, 0,
1, 1, 0, 0,

instead of a rectar scan,

I I want to group just 0, i should  ((0))
However, if i want to group (0, 0), i should have ((0), (0))

For reshaping with pixel shufflign:

one grouping couild be

0 0 
2 0

from

X, 0, X, 2
1, 1, 0, 0,
X, 2, X, 0,
1, 1, 0, 0,


and 

0 2 
2 0


X, Y, X, Y
1, 1, 0, 0,
X, Y, X, Y,
1, 1, 0, 0,

This means I could create 4 groupings of 2x2 from the 4x4 image

e.g. make the remaining numbers follow the square pattern grouping

Tuples:
the tuple of the first group with be ((0, 2, 2, 0)), do this via reshaping numpy

BPE Is multidimensional, so its still "adjacent"
the first group and the second group are still adjacent.

(("0","0")) should never exist, but (("0"),("0")) should exist, for example


First: make all single numbers into tuples. Reshape it to a dividable tuple.
The dictionary should also be tuples of tuples. (Should be consistent)

Second: BPE should have low time complexity. Perhaps use the mini bpe

"""


def flatten_data(images):
    flattened = []
    for image in images:
        for row in image_to_list_of_lists(image):
            flattened.append(row)
    return flattened


def apply_merges_to_row(row, merges):
    new_row = []
    skip_next = False
    for i in range(len(row)):
        if skip_next:
            skip_next = False
            continue
        if i < len(row) - 1 and (row[i], row[i + 1]) in merges:
            new_row.append(merges[(row[i], row[i + 1])])
            skip_next = True
        else:
            new_row.append(row[i])
    return new_row


def perform_bpe_on_dataset(flattened_data, num_merges):
    merges = {}
    for _ in range(num_merges):
        pair_freqs = Counter()
        for row in flattened_data:
            for i in range(len(row) - 1):
                pair = (row[i], row[i + 1])
                pair_freqs[pair] += 1

        if not pair_freqs:
            break

        most_common_pair = pair_freqs.most_common(1)[0][0]
        new_token = most_common_pair
        merges[most_common_pair] = new_token

        # apply the merge across the dataset
        for i in range(len(flattened_data)):
            flattened_data[i] = apply_merges_to_row(flattened_data[i], merges)

    return flattened_data, merges


N = 1
flattened_data = flatten_data(x_train[:N])
num_merges = 50
processed_data, merges_performed = perform_bpe_on_dataset(flattened_data, num_merges)

# build vocab
token_counts = Counter()
for row in processed_data:
    token_counts.update(row)
for token, count in token_counts.items():
    if isinstance(token, tuple):
        print(f"{token}: {count}")
    else:
        print(f"('{token}'): {count}")
