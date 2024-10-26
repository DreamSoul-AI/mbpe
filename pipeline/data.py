import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from mbpe import utils

class MNISTDataset(Dataset):
    def __init__(self, mnist_data, dim):
        self.mnist_data = mnist_data
        self.dim = dim

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        image, _ = self.mnist_data[idx]
        shuffled_image = utils.pixel_to_tuples(image, dim=self.dim)
        return shuffled_image
    
def custom_collate_fn(batch):
    return [item for item in batch[0]]
    
def load_mnist_dataset(config, batch_size=1, train=True):
    """
    Load the MNIST dataset and create a DataLoader for it.

    Args:
        batch_size: Batch size for DataLoader
        train: Whether to load the training or test set

    Returns:
        DataLoader for the MNIST dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x * 255).to(dtype=torch.uint8)
    ])
    mnist_dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    
    dataset = MNISTDataset(mnist_dataset, config.dim)
    if config.size > 0:
        dataset = Subset(dataset, range(config.size))
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, collate_fn=custom_collate_fn)
    
    return data_loader
