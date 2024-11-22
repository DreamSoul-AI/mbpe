import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import mbpe

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
    train_dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    tokenizer = mbpe.tokenizer.Tokenizer()
    output_ids = []

    # vocab_len = tokenizer.get_vocab_len()
    # progress = tqdm(train_loader, desc=f"vocab size [{vocab_len}]")

    max_shape = (1, 2, 2)
    dim_index = [1, 2, 3]
    min_freq = 2
    root_min_freq = 2
    for image, label in train_loader:
        # data = mbpe.utils.tensor_to_tuples(data, dim, dim_index)
        tokenizer.train(image, max_shape=max_shape, min_freq=min_freq, root_min_freq=root_min_freq)
        # tokenized = tokenizer.encode(data, dim=dim)
        # output_ids.append(tokenized)
        break
    # vocab_len = tokenizer.get_vocab_len()
    print('vocabulary:', tokenizer.get_vocab())
    # print('output_ids:', output_ids)
