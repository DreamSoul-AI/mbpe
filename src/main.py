import torch
import torchvision
import torchvision.transforms as transforms
import mbpe
from torch.utils.data import Subset


if __name__ == "__main__":
    torch.manual_seed(1)
    batch_size = 2
    max_workers = 0
    data_name = 0
    max_code_size = (1, 2, 2)
    dim_index = [2, 3, 4]
    min_freq = {'root': 0.01, 'merge': 0.01, 'entry': 0.005}

    transform = transforms.Compose([
        transforms.ToTensor(),
        mbpe.utils.AddSequenceDim(0),
        lambda x: (x * 255).to(dtype=torch.uint8)
    ])

    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    indices = list(range(0, 2))  # load only the first n samples # TODO: not enough images
    dataset = Subset(dataset, indices)


    tokenizer = mbpe.tokenizer.Tokenizer(batch_size=batch_size, max_workers=max_workers)
    tokenizer.train(dataset, data_name, max_code_size, dim_index, min_freq)
    print('vocabulary:', tokenizer.vocab)
