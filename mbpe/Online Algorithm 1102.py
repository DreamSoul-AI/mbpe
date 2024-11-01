from utils import *
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
from collections import Counter
from itertools import chain
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
random_seed = 2

transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: (x * 255).to(dtype=torch.uint8)
])

torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

batch_size = 1
random_seed = 1

train_dataset = torchvision.datasets.MNIST(root='D:\\PythonProjects\\JupyterNotebooks\\PixelLevelBPE\\data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
progress = tqdm(train_loader)

tuple_list_2_2 = []
for i, (image, label) in enumerate(progress):
    if i == 10:
        break
    tuple_list = reshape_to_tuples(image, dim=(2, 2))
    tuple_list_2_2.append(tuple_list)

tuple_list_2_2 = list(chain.from_iterable(tuple_list_2_2))

## 函数
# 新增
def get_freq(tuples_list, frequency_dict):
    global total_count
    for t in tuples_list:
        if isinstance(t, tuple):
            frequency_dict[t] += 1
            total_count += 1

# 新增
def deep_compare(a, b):
    if type(a) != type(b):
        return False

    try:
        if len(a) != len(b):
            return False
        return all(deep_compare(x, y) for x, y in zip(a, b))

    except TypeError:
        pass
    return a == b

# 有修改
def merge(ids, pair):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and deep_compare(ids[i], pair[0]) and deep_compare(ids[i + 1], pair[1]):
            new_ids.append(pair)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# 新增 encode
def replace_with_codes(item, root_vocab_with_codes):
    if item in root_vocab_with_codes:
        return root_vocab_with_codes[item]
    else:
        return item

# 新增
def build_vocabulary(tuple_list_dim, vocab_size, root_vocab):
    for _ in range(vocab_size):
        stats = freq_pair(tuple_list_dim)
        pair = max(stats, key=stats.get)
        tuple_list_dim = merge(tuple_list_dim, pair)
        root_vocab.add(pair)

        if len(root_vocab) >= vocab_size:
            break
    return root_vocab, tuple_list_dim

## dim (2, 2)
frequency_dict_2_2 = defaultdict(int)
total_count = 0
root_vocab_2_2 = set()

get_freq(tuple_list_2_2, frequency_dict_2_2)

for tuple_item, count in frequency_dict_2_2.items():
    frequency = count / total_count
    if frequency > 0.002:
        root_vocab_2_2.add(tuple_item)

print("Final Tuple Counts and Frequencies:")
for tuple_item, count in frequency_dict_2_2.items():
    frequency = count / total_count
    print(f"Tuple: {tuple_item}, Count: {count}, Frequency: {frequency:.4f}")


print("\nRoot Vocabulary (Frequency > 0.002):")
for tuple_item in root_vocab_2_2:
    print(tuple_item)

print(f"\nTotal tuples processed: {total_count}")

root_vocab_2_2, tuple_list_2_2_merged = build_vocabulary(tuple_list_2_2, 100, root_vocab_2_2)
root_vocab_with_codes_2_2 = {element: idx for idx, element in enumerate(root_vocab_2_2, start=1)}
encoded_list_2_2 = [replace_with_codes(item, root_vocab_with_codes_2_2) for item in tuple_list_2_2_merged]


## dim (2, 1)
tuple_list_2_1 = reshape_tuple(encoded_list_2_2)
frequency_dict_2_1 = defaultdict(int)
total_count = 0
root_vocab_2_1 = set()

get_freq(tuple_list_2_1, frequency_dict_2_1)

for tuple_item, count in frequency_dict_2_1.items():
    frequency = count / total_count
    if frequency > 0.002:
        root_vocab_2_1.add(tuple_item)

print("Final Tuple Counts and Frequencies:")
for tuple_item, count in frequency_dict_2_1.items():
    frequency = count / total_count
    print(f"Tuple: {tuple_item}, Count: {count}, Frequency: {frequency:.4f}")


print("\nRoot Vocabulary (Frequency > 0.002):")
for tuple_item in root_vocab_2_1:
    print(tuple_item)

print(f"\nTotal tuples processed: {total_count}")

root_vocab_2_1, tuple_list_2_1_merged = build_vocabulary(tuple_list_2_1, 150, root_vocab_2_1)
root_vocab_with_codes_2_1 = {element: idx for idx, element in enumerate(root_vocab_2_1, start=len(root_vocab_with_codes) + 1)}
encoded_list_2_1 = [replace_with_codes(item, root_vocab_with_codes_2_1) for item in tuple_list_2_1_merged]
