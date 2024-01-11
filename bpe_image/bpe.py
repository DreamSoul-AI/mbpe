import numpy as np
from torchvision import datasets, transforms
from transformers import BertTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from collections import Counter

# Load MNIST 
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
subset_size = 100
subset_images = mnist_train.data[:subset_size].numpy()

# Reshape subset to (100, 28*28) and convert to binary strings
reshaped_subset = subset_images.reshape(subset_size, -1)
binary_strings = [''.join(map(str, image)) for image in reshaped_subset]

# BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

# Train
trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
tokenizer.train_from_iterator(binary_strings, trainer=trainer)
vocab_dict = tokenizer.get_vocab()
for key, value in vocab_dict.items():
    print(f"Key: {key}, Value: {value}")