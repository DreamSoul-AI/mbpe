from tensorflow.keras.datasets import mnist
import numpy as np
from collections import defaultdict, Counter

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def image_to_list_of_lists(image):
    return [[str(pixel) for pixel in row] for row in image]

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
        if i < len(row) - 1 and (row[i], row[i+1]) in merges:
            new_row.append(merges[(row[i], row[i+1])])
            skip_next = True
        else:
            new_row.append(row[i])
    return new_row

def perform_bpe_on_dataset(flattened_data, num_merges):
    merges = {}
    merge_token_counter = 0  # Counter to create unique merge identifiers
    merge_history = {}  # Dictionary to track the history of each merge

    for _ in range(num_merges):
        pair_freqs = Counter()
        for row in flattened_data:
            for i in range(len(row) - 1):
                pair = (row[i], row[i+1])
                pair_freqs[pair] += 1

        if not pair_freqs:
            break

        most_common_pair = pair_freqs.most_common(1)[0][0]
        merge_token = f"merge_{merge_token_counter}"
        merge_token_counter += 1

        # Prepare components for merge_history
        components = [most_common_pair[0], most_common_pair[1]]
        # Check if components are themselves merges and adjust representation
        components = [merge_history.get(component, component) for component in components]
        # Update merge_history to include the new merge
        merge_history[merge_token] = tuple(components)

        # Apply the merge across the dataset
        for i in range(len(flattened_data)):
            new_row = []
            skip_next = False
            for j in range(len(flattened_data[i])):
                if skip_next:
                    skip_next = False
                    continue
                if j < len(flattened_data[i]) - 1 and (flattened_data[i][j], flattened_data[i][j+1]) == most_common_pair:
                    new_row.append(merge_token)
                    skip_next = True
                else:
                    new_row.append(flattened_data[i][j])
            flattened_data[i] = new_row

    return flattened_data, merge_history

N = 1
flattened_data = flatten_data(x_train[:N])
num_merges = 50
processed_data, merge_history = perform_bpe_on_dataset(flattened_data, num_merges)

# Build and print the vocabulary with counts
token_counts = Counter()
for row in processed_data:
    token_counts.update(row)

for token, count in token_counts.items():
    print(f"{token}: {count}")

# Print merge history
print("\nMerge history:")
for merge_token, components in merge_history.items():
    print(f"{merge_token} represents: {components}")