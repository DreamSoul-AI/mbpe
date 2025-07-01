import torch
import torchvision
import torchvision.transforms as transforms
import mbpe
from torch.utils.data import Subset


def main():
    torch.manual_seed(1)
    batch_size = 1
    max_workers = 0
    data_name = 0
    num_samples = 1
    max_codeword_size = (1, 2, 2)
    dim_index = [1, 2, 3]
    min_freq = {'root': 0.01, 'merge': 0.001, 'freq_counter': 0.01}  # TODO: take in int

    transform = transforms.Compose([
        transforms.ToTensor(),
        mbpe.utils.AddSequenceDim(0),
        lambda x: (x * 255).to(dtype=torch.uint8)
    ])

    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    if num_samples > 0:
        indices = list(range(0, num_samples))
        dataset = Subset(dataset, indices)

    sample = dataset[0][data_name]
    print('sample:', sample)

    tokenizer = mbpe.tokenizer.Tokenizer(min_freq, batch_size=batch_size, max_workers=max_workers)
    tokenizer.train(dataset, data_name, max_codeword_size, dim_index)
    print('vocabulary:', tokenizer.vocab)
    encoded = tokenizer.encode(sample, max_codeword_size=max_codeword_size, dim_index=dim_index)
    print("encoded:", encoded)
    exit()
    decoded = tokenizer.decode(encoded, max_codeword_size=max_codeword_size, dim_index=dim_index,
                               data_shape=sample.shape, data_dtype=sample.dtype)

    print("Decoded shape:", decoded.shape)
    print("Exact match:", torch.equal(decoded, sample))
    return


if __name__ == "__main__":
    main()
