import torch
import torchvision
import torchvision.transforms as transforms
import mbpe
from torch.utils.data import Subset


def main():
    torch.manual_seed(1)
    batch_size = 4
    max_workers = 0
    data_name = 0
    num_samples = 0
    max_codeword_size = (1, 2, 2)
    dim_index = [1, 2, 3]
    min_freq = {'root': 0.01, 'merge': 0.001, 'freq_counter': 0.01}  # TODO: take in int, can give known root as init

    transform = transforms.Compose([
        transforms.ToTensor(),
        # mbpe.utils.AddSequenceDim(0),
        lambda x: (x * 255).to(dtype=torch.uint8)
    ])

    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    if num_samples > 0:
        indices = list(range(0, num_samples))
        dataset = Subset(dataset, indices)

    tokenizer = mbpe.tokenizer.Tokenizer(min_freq, max_codeword_size=max_codeword_size, batch_size=batch_size,
                                         max_workers=max_workers)
    tokenizer.train(dataset, data_name)
    print('vocabulary:', tokenizer.vocab)
    # exit()

    # sample = dataset[2][data_name].unsqueeze(0) # first dimension as seq
    # print('sample:', sample.size(), sample)
    # print(tokenizer.encode(sample))
    # exit()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    encoded = []
    for batch in loader:
        encoded.extend(tokenizer.encode(batch[data_name]))
        break
    print('encoded:', encoded)
    exit()

    decoded = tokenizer.decode(encoded, data_shape=sample.shape, data_dtype=sample.dtype)
    print("Decoded shape:", decoded.shape)
    if torch.allclose(decoded, sample):
        print("✅ Decoded output matches original input.")
    else:
        print("❌ Decoded output not match original input.")
    exit()
    sort_sample = torch.tensor(sorted(sample.view(-1).tolist()))
    sort_decoded = torch.tensor(sorted(decoded.view(-1).tolist()))
    # print(sort_sample)
    # print(sort_decoded)
    # print("Sorted match:", torch.equal(sort_decoded, sort_sample))
    return


if __name__ == "__main__":
    main()
