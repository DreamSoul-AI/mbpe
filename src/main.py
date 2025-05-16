import torch
import torchvision
import torchvision.transforms as transforms
import mbpe
from torch.utils.data import Subset


def main():
    torch.manual_seed(1)
    batch_size = 2
    max_workers = 0
    data_name = 0
    num_samples = 10
    max_codeword_size = (1, 2, 2)
    dim_index = [1, 2, 3]
    min_freq = {'root': 0.01, 'merge': 0.01, 'freq_counter': 0.001}

    transform = transforms.Compose([
        transforms.ToTensor(),
        mbpe.utils.AddSequenceDim(0),
        lambda x: (x * 255).to(dtype=torch.uint8)
    ])

    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    if num_samples > 0:
        indices = list(range(0, num_samples))
        dataset = Subset(dataset, indices)

    tokenizer = mbpe.tokenizer.Tokenizer(min_freq, batch_size=batch_size, max_workers=max_workers)
    tokenizer.train(dataset, data_name, max_codeword_size, dim_index)
    print('vocabulary:', tokenizer.vocab)
    return


if __name__ == "__main__":
    main()
