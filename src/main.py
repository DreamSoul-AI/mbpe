import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import mbpe

torch.manual_seed(1)
batch_size = 1

transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: (x * 255).to(dtype=torch.uint8)
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    tokenizer = mbpe.tokenizer.Tokenizer(max_workers=batch_size)
    output_ids = []

    vocab_len = len(tokenizer)
    # progress = tqdm(train_loader, desc=f"vocab size [{vocab_len}]")

    max_shape = (1, 2, 2)
    dim_index = [2, 3, 4]
    min_freq = 2
    root_min_freq = 2
    for image, label in train_loader:
        tokenizer.train(image, max_shape, dim_index, min_freq, root_min_freq)
        # tokenized = tokenizer.encode(data, dim=dim)
        # output_ids.append(tokenized)
        # current_vocab_size = len(tokenizer)
        # progress.set_description(f"Vocab Size: {current_vocab_size}")
        break
    print('vocabulary:', tokenizer.get_vocab())
    # print('output_ids:', output_ids)
