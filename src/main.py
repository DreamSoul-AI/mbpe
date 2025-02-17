import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import mbpe

torch.manual_seed(1)
batch_size = 1
max_workers = 0

transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: (x * 255).to(dtype=torch.uint8).unsqueeze(0)
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
indices = list(range(0, 2))  # load only the first n samples
subset_dataset = Subset(train_dataset, indices)
train_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    tokenizer = mbpe.tokenizer.Tokenizer(max_workers=max_workers)

    max_shape = (1, 2, 2)
    dim_index = [2, 3, 4]
    min_freq = 0.01
    root_min_freq = 0.01
    tokenizer.train(train_loader, max_shape, dim_index, min_freq, root_min_freq)
    print('vocabulary:', tokenizer.get_vocab())
