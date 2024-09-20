from utils import *
import numpy as np
from collections import defaultdict
import pickle
from collections import Counter
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import os
import pickle
random_seed = 2

transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: (x * 255).to(dtype=torch.uint8)
])

torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

batch_size = 1
random_seed = 1

train_dataset = torchvision.datasets.MNIST(root='D:\\PythonProjects\\JupyterNotebooks\\PixelLevelBPE\\data', train=True,
                                           download=True, transform=transform)  # 下载并加载 MNIST 训练数据集，应用上述转换。
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
progress = tqdm(train_loader)

tuple_list_2_2 = []

for i, (image, label) in enumerate(progress):
    if i == 10:
        break
    tuple_list = reshape_to_tuples(image, dim=(2, 2))
    tuple_list_2_2.append(tuple_list)

# ----------------------------------------------------------------------------------
class FrequencyTableManager:
    def __init__(self, freq_file='freq_tab_2_2.pkl', prob_file='prob_tab_2_2.pkl'):
        self.freq_file = freq_file
        self.prob_file = prob_file
        self.freq_tab = {}
        self.prob_tab = {}

        self._initialize_tables()

    def _initialize_tables(self):

        # Initialize frequency file
        if not os.path.exists(self.freq_file):
            with open(self.freq_file, 'wb') as f:
                pickle.dump({}, f)
        else:
            with open(self.freq_file, 'rb') as f:
                self.freq_tab = pickle.load(f)

        # Initialize probability file
        if not os.path.exists(self.prob_file):
            with open(self.prob_file, 'wb') as f:
                pickle.dump({}, f)
        else:
            with open(self.prob_file, 'rb') as f:
                self.prob_tab = pickle.load(f)

    def update_tables(self, tuple_list):
        for tup in tuple_list:
            self._update_frequency(tup)
            self._update_probability()

    def _update_frequency(self, tup):
        # Update frequency table
        if tup in self.freq_tab:
            self.freq_tab[tup] += 1
        else:
            self.freq_tab[tup] = 1

        # Save frequency table
        with open(self.freq_file, 'wb') as f:
            pickle.dump(self.freq_tab, f)

    def _update_probability(self):

        total_count = sum(self.freq_tab.values())

        # Update probability table
        self.prob_tab = {tup: count / total_count for tup, count in self.freq_tab.items()}

        # Save probability table
        with open(self.prob_file, 'wb') as f:
            pickle.dump(self.prob_tab, f)

    def get_freq_tab(self):
        return self.freq_tab

    def get_prob_tab(self):
        return self.prob_tab


manager = FrequencyTableManager()

for image_list in tuple_list_2_2:
    manager.update_tables(image_list)

freq_tab = manager.get_freq_tab()
prob_tab = manager.get_prob_tab()

print("Frequency table:", freq_tab)
print("Probability table:", prob_tab)
