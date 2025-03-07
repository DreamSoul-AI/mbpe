import torch
import torchvision
import torchvision.transforms as transforms
import mbpe
from torch.utils.data import Subset

torch.manual_seed(1)
batch_size = 1
max_workers = 0

transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: (x * 255).to(dtype=torch.uint8).unsqueeze(0)
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
indices = list(range(0, 2))  # load only the first n samples # TODO: not enough images
dataset = Subset(dataset, indices)

if __name__ == "__main__":
    tokenizer = mbpe.tokenizer.Tokenizer(batch_size=batch_size, max_workers=max_workers)

    data_name = 0
    max_shape = (1, 2, 2)
    dim_index = [2, 3, 4]
    min_freq = 0.01
    root_min_freq = 0.01
    min_entrance_freq = 0.01
    tokenizer.train(dataset, data_name, max_shape, dim_index, min_freq, root_min_freq, min_entrance_freq)
    print('vocabulary:', tokenizer.get_vocab())
