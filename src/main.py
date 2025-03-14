import torch
import torchvision
import torchvision.transforms as transforms
import mbpe
from torch.utils.data import Subset


transform = transforms.Compose([
    transforms.ToTensor(),
    mbpe.utils.AddSequenceDim(0),
    lambda x: (x * 255).to(dtype=torch.uint8)
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
indices = list(range(0, 2))  # load only the first n samples # TODO: not enough images
dataset = Subset(dataset, indices)

if __name__ == "__main__":
    torch.manual_seed(1)
    batch_size = 2
    max_workers = 0
    data_name = 0
    max_code_size = (1, 2, 2)
    dim_index = [2, 3, 4]
    min_freq = 0.01
    root_min_freq = 0.01
    min_entrance_freq = 0.01

    tokenizer = mbpe.tokenizer.Tokenizer(batch_size=batch_size, max_workers=max_workers)
    tokenizer.train(dataset, data_name, max_code_size, dim_index, min_freq, root_min_freq, min_entrance_freq)
    print('vocabulary:', tokenizer.get_vocab())
