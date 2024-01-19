from tokenizers import ByteLevelBPETokenizer
from collections import Counter
import numpy as np
from numpy import asarray
import torch
from PIL import Image
from torchvision import datasets, transforms

# first image of MNIST, change the path according to your own images
image = Image.open('0.jpg')

# summarize some details about the image
# print(image.format)
# print(image.size)
# print(image.mode)

# asarray() class is used to convert
# PIL images into NumPy arrays
img_np = asarray(image)
# print(img_np)
# print(type(img_np))
# print(img_np.shape)

img_np_flat = img_np.flatten()
# print(img_np_flat)
# print(img_np_flat.shape)

img_str = [str(i) for i in img_np_flat]
# print(img_str)

# Define a transform to convert PIL
# image to a Torch tensor
# transform = transforms.Compose([transforms.ToTensor()])

# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
# img_tensor = transform(image)
# print(img_tensor.size())

# Train BPE
tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(img_str)

# Get vocabulary
token_counts = Counter(tokenizer.get_vocab())
with open('vocab.txt', 'a') as record:
    for token, count in token_counts.most_common():
        print(f"Token: {token}, Count: {count}", file=record)
