"""Gray Scale Image BPE (batch_size = 1) """

from collections import defaultdict
from numpy import asarray
from PIL import Image

# first image of MNIST, change the path according to your own images
image = Image.open('./0.jpg')
# summarize some details about the image
# print(image.format)
# print(image.size)
# print(image.mode)


def convert_image_to_string_list(single_image):
    # asarray() class is used to convert
    # PIL images into NumPy arrays
    img_np = asarray(single_image)
    # print(img_np)
    # print(type(img_np))
    # print(img_np.shape)
    img_np_flat = img_np.flatten()
    # print(img_np_flat)
    # print(img_np_flat.shape)
    # print(type(img_np_flat))
    img_string = [str(i) for i in img_np_flat]
    print("\nInitial image strings: ", img_string)
    return img_string


img_str = convert_image_to_string_list(image)

str_freqs = defaultdict(int)
for string in img_str:
    str_freqs[string] += 1

print("String frequencies: ", str_freqs)

alphabet = []

for string in str_freqs.keys():
    if string not in alphabet:
        alphabet.append(string)
# alphabet.sort()

print("\nInitial vocabulary: ", alphabet)
print("Initial vocabulary length: ", len(alphabet), "\n")
vocab = alphabet.copy()


def compute_pair_freqs(strings):
    str_pair_freqs = defaultdict(int)
    for i in range(len(strings) - 1):
        str_pair = (strings[i], strings[i + 1])
        str_pair_freqs[str_pair] += 1
    return str_pair_freqs


def merge_pair(a, b, split):
    i = 0
    while i < len(split) - 1:
        if split[i] == a and split[i + 1] == b:
            split = split[:i] + [a + b] + split[i + 2:]
        else:
            i += 1
    return split


vocab_size = 10000
freq_threshold = 2

print("The merging process: ")
merges = {}
while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(img_str)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    if max_freq < freq_threshold:
        break
    print("current best_pair, current max_freq: ", best_pair, max_freq)
    img_str = merge_pair(*best_pair, img_str)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

print("\nMerged strings of a gray image: ", img_str)
print("\nmerges (will be used when inference): ", merges)
print("The vocabulary: ", vocab)
print("The length of vocabulary: ", len(vocab))


def tokenize(str_list):
    for str_pair, merge in merges.items():
        i = 0
        while i < len(str_list) - 1:
            if str_list[i] == str_pair[0] and str_list[i + 1] == str_pair[1]:
                str_list = str_list[:i] + [merge] + str_list[i + 2:]
            else:
                i += 1

    return str_list


test_image = Image.open('./1.jpg')
test_list = convert_image_to_string_list(test_image)
test_merged_list = tokenize(test_list)
print("test_merged_list: ", test_merged_list)
