from collections import defaultdict
from keras.datasets import mnist
import numpy as np

def get_binary_representation(x):
    binary_x = []
    for image in x:
        binary_image = np.unpackbits(image).reshape(-1, 8)  # Convert pixel values to binary representation
        binary_image = binary_image.astype(str)  # Convert the binary values to strings
        binary_x.append(binary_image)
    return binary_x

def get_vocabulary(texts):
    vocabulary = defaultdict(int)
    for text in texts:
        for token in text:
            token_tuple = tuple(token)
            vocabulary[token_tuple] += 1
    return vocabulary

def merge_tokens(vocabulary, merge_pair):
    new_vocabulary = defaultdict(int)
    for token, freq in vocabulary.items():
        token_str = ' '.join(token)
        new_token_str = token_str.replace(merge_pair, merge_pair.replace(" ", ""))
        new_token = tuple(new_token_str.split(' '))
        new_vocabulary[new_token] = freq
    return new_vocabulary

def bpe(texts, num_merges):
    vocabulary = get_vocabulary(texts)
    updated_texts = []

    for _ in range(num_merges):
        pairs = defaultdict(int)
        for text in texts:
            for row in text:
                row_str = ' '.join(row)
                for i in range(len(row) - 1):
                    pair = row[i] + " " + row[i+1]
                    pairs[pair] += 1

        most_common_pair = max(pairs, key=pairs.get)
        vocabulary = merge_tokens(vocabulary, most_common_pair)

        updated_texts.clear()  # Clear the list for the new iteration

        for text in texts:
            updated_text = []
            for row in text:
                row_str = ' '.join(row)
                updated_row_str = row_str.replace(most_common_pair, most_common_pair.replace(" ", ""))
                updated_row = updated_row_str.split(' ')
                updated_text.append(np.array(updated_row))
            updated_texts.append(updated_text)

        texts = updated_texts.copy()  # Update texts for the next iteration

    return vocabulary, updated_texts


# Load the MNIST dataset
(x_train, y_train), _ = mnist.load_data()

# Convert pixel values to binary representation
binary_x_train = get_binary_representation(x_train[0])

# Run BPE
num_merges = 3
vocabulary = bpe(binary_x_train, num_merges)
sorted_vocabulary = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)

print("Vocabulary after BPE:")
for token, freq in sorted_vocabulary:
    print(f"{token}: {freq}")
