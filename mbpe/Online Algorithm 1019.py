from utils import *
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
from collections import Counter
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


frequency_dict = defaultdict(int)
total_count = 0
root_vocab = {}

def get_freq(tuples_list):
    global total_count
    for t in tuples_list:
        frequency_dict[t] += 1
        total_count += 1


for image in tuple_list_2_2:
    get_freq(image)

for tuple_item, count in frequency_dict.items():
    frequency = count / total_count
    if frequency > 0.002:
        root_vocab[tuple_item] = frequency

# output count & frequency
print("Final Tuple Counts and Frequencies:")
for tuple_item, count in frequency_dict.items():
    frequency = count / total_count
    print(f"Tuple: {tuple_item}, Count: {count}, Frequency: {frequency:.4f}")

# output root_vocab (frequency > 0.002)
print("\nRoot Vocabulary (Frequency > 0.002):")
for tuple_item, frequency in root_vocab.items():
    print(f"Tuple: {tuple_item}, Frequency: {frequency:.4f}")

print(f"Total tuples processed: {total_count}")