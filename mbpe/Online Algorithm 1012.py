from utils import *
import numpy as np
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


# Record count table & frequency table
try:
    with open('count_tab_2_2.pkl', 'rb') as f:
        count_tab_2_2 = pickle.load(f)
except FileNotFoundError:
    count_tab_2_2 = defaultdict(int)

for image_list in tuple_list_2_2:
    for tup in image_list:
        count_tab_2_2[tup] += 1

    total_count = sum(count_tab_2_2.values())

    freq_tab_2_2 = {tup: count / total_count for tup, count in count_tab_2_2.items()}

    with open('count_tab_2_2.pkl', 'wb') as f:
        pickle.dump(dict(count_tab_2_2), f)

    with open('freq_tab_2_2.pkl', 'wb') as f:
        pickle.dump(freq_tab_2_2, f)