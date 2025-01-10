import os
from frequency_counter import FrequencyCounter
from utils import *

from itertools import islice
import tqdm
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

train_dataset = torchvision.datasets.MNIST(root='C:\\Users\\YOGO\\mbpe\\data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

progress = tqdm.tqdm(train_loader)

tuple_list_2_2 = []

for i, (image, label) in enumerate(progress):
    if i == 10:
        break
    tuple_list = reshape_to_tuples(image, dim=(2, 2))
    tuple_list_2_2.append(tuple_list)

freq_file_path = "data/output/root_frequency_file_2_2.pkl"
vocab_file_path = "data/output/root_vocabulary_file_2_2.pkl"

os.makedirs(os.path.dirname(freq_file_path), exist_ok=True)

counter = FrequencyCounter()

reset_flag = True

for tuple_list in tuple_list_2_2:
    root_frequency, root_vocab = counter.calculate_and_filter_frequency(
        input_list=tuple_list,
        freq_file_path=freq_file_path,
        vocab_file_path=vocab_file_path,
        reset=reset_flag,
        min_root_freq=0.01
    )
    reset_flag = False

print('Root vocabulary for dim (2, 2):', root_vocab)
