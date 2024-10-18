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
    train_dataset, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    tokenizer = mbpe.tokenizer.Tokenizer()
    tokenized_tuples = []

    vocab_len = tokenizer.get_vocab_len()
    # progress = tqdm(train_loader, desc=f"vocab size [{vocab_len}]")
    progress = train_loader

    dim = (1, 2, 2)
    dim_index = [1, 2, 3]
    for image, label in progress:
        tuples = mbpe.utils.tensor_to_tuples(image, dim, dim_index)
        print(tuples)
        exit()
        tokenized = tokenizer.train(tuples, dim_index, dim=dim, min_freq=2)
        tokenized_tuples.append(tokenized)

        vocab_len = tokenizer.get_vocab_len()
        progress.set_description(f"vocab size [{vocab_len}]")

    print('vocabulary:', tokenizer.get_vocab())
    print('tokenized_tuples:', tokenized_tuples)
