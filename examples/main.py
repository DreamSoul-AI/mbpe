import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import sys
import os
from tqdm import tqdm

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from mbpe import basic


batch_size = 1
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: (x * 255).to(dtype=torch.uint8)
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    tokenizer = basic.BasicTokenizer()
    tokenized_tuples = []

    vocab_len = tokenizer.get_vocab_len()
    progress = tqdm(train_loader, desc=f"vocab size [{vocab_len}]")
    for image, label in progress:
        tokenized = tokenizer.train_encode(image, 12000, min_freq=4)
        tokenized_tuples.append(tokenized)

        vocab_len = tokenizer.get_vocab_len()
        progress.set_description(f"vocab size [{vocab_len}]")

    print('vocabulary:', tokenizer.get_vocab())