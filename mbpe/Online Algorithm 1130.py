from utils import *
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
from collections import Counter
from itertools import chain
import tensorflow as tf
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


# 新增
def tuple_freq(tuple_list, min_root_freq):
    frequency_dict = {}
    total_tuples = 0

    # 统计元组出现次数并计算总元组数
    for item in tuple_list:
        if isinstance(item, tuple):  # 仅处理元组
            total_tuples += 1
            if item in frequency_dict:
                frequency_dict[item] += 1
            else:
                frequency_dict[item] = 1

    # 计算频率并筛选高频元组
    root_vocab = set()
    for tuple_item, count in frequency_dict.items():
        frequency = count / total_tuples
        frequency_dict[tuple_item] = frequency  # 更新为频率
        if frequency > min_root_freq:
            root_vocab.add(tuple_item)

    return root_vocab, frequency_dict


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
def build_vocabulary(tuple_list_dim, min_freq, root_vocab):
    while True:
        stats = freq_pair(tuple_list_dim)

        if not stats:
            break

        pair = max(stats, key=stats.get)
        pair_freq = stats[pair] / sum(stats.values())

        if pair_freq > min_freq:
            root_vocab.add(pair)
            tuple_list_dim = merge(tuple_list_dim, pair)
        else:
            break

    return root_vocab, tuple_list_dim

def process_nested_tuples(tuple_list, min_freq, root_vocab=None, start_code=1):
    vocab, tuple_list_merged = build_vocabulary(tuple_list, min_freq, root_vocab)
    vocab_with_codes = {item: idx for idx, item in enumerate(vocab, start=start_code)}
    encoded_list = [replace_with_codes(item, vocab_with_codes) for item in tuple_list_merged]
    return vocab, tuple_list_merged, encoded_list, vocab_with_codes

def process_all_levels(initial_tuple_list, min_freq, min_root_freq):
    vocab_offset = 1
    results = []

    # 第一级处理
    root_vocab, frequency_dict = tuple_freq(initial_tuple_list, min_root_freq)
    vocab, tuple_list_merged, encoded_list, vocab_with_codes = process_nested_tuples(
        initial_tuple_list, min_freq, root_vocab = root_vocab, start_code=vocab_offset
    )
    vocab_offset += len(vocab)
    results.append({
        "tuple_list_merged": tuple_list_merged,
        "encoded_list": encoded_list,
        "vocab_with_codes": vocab_with_codes,
    })

    reshaped_tuple_list = reshape_tuple(encoded_list)
    root_vocab, frequency_dict = tuple_freq(reshaped_tuple_list, min_root_freq)
    vocab, tuple_list_merged, encoded_list, vocab_with_codes = process_nested_tuples(
        reshaped_tuple_list, min_freq, root_vocab = root_vocab, start_code=vocab_offset
    )
    vocab_offset += len(vocab)
    results.append({
        "tuple_list_merged": tuple_list_merged,
        "encoded_list": encoded_list,
        "vocab_with_codes": vocab_with_codes,
    })

    reshaped_tuple_list = reshape_tuple(encoded_list)
    root_vocab = {(i,) for i in range(256)}
    vocab, tuple_list_merged, encoded_list, vocab_with_codes = process_nested_tuples(
        reshaped_tuple_list, min_freq, root_vocab = root_vocab, start_code=vocab_offset
    )
    vocab_offset += len(root_vocab)
    results.append({
        "tuple_list_merged": tuple_list_merged,
        "encoded_list": encoded_list,
        "vocab_with_codes": vocab_with_codes,
    })

    return encoded_list


# Define min_root_freq values for 10 plots
min_root_freq_values = np.linspace(0.01, 0.1, 10)
min_freq_values = np.linspace(0.001, 0.1, 50)

# Loop through each min_root_freq and generate a plot
for idx, min_root_freq in enumerate(min_root_freq_values, start=1):
    compression_ratio = []
    for min_freq in min_freq_values:
        encoded_list = process_all_levels(tuple_list_2_2, min_freq, min_root_freq)
        compression_ratio.append(len(encoded_list) / (len(tuple_list_2_2) * 4))

    # Create the line chart
    plt.figure(figsize=(10, 6))
    plt.plot(min_freq_values, compression_ratio, marker='o', linestyle='-', color='b')
    plt.xlabel('min_freq')
    plt.ylabel('compression_ratio')
    plt.title(f'Compression Ratio Variation with min_freq (min_root_freq={min_root_freq:.2f})')
    plt.grid(True)

    # Save the plot as an image
    plt.savefig(f"compression_ratio_plot_{idx}.png")
    plt.close()

print("All 10 plots have been generated and saved as images.")