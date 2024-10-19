import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from config import Config
from data import load_mnist_dataset
from model import get_model
from mbpe_tokenizer import tokenize

def train():
    config = Config()
    model = get_model(config)

    train_loader = load_mnist_dataset(config)

    dataset = Dataset.from_dict(tokenize(train_loader, config, verbose=False))
    print(dataset)
    
    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     num_train_epochs=config.num_train_epochs,
    #     per_device_train_batch_size=config.batch_size,
    #     weight_decay=config.weight_decay,
    #     logging_dir="./logs",
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=,
    # )

    # trainer.train()

if __name__ == "__main__":
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    train()