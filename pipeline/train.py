import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from config import Config
from data import load_mnist_dataset
from model import get_model
from mbpe import tokenizer
from tqdm import tqdm

def pad_sequences(sequences, pad_token_id=0, max_length=None):
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        padded_sequences = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)

        for i, seq in enumerate(sequences):
            seq = [int(x) for x in seq]
            end = min(len(seq), max_length)
            padded_sequences[i, :end] = torch.tensor(seq[:end], dtype=torch.long)
            attention_mask[i, :end] = 1

        return padded_sequences, attention_mask

def train_tokenizer(input_image, mbpe_tokenizer, config, verbose=False):
    outputs = []
    vocab_len = mbpe_tokenizer.get_vocab_len()
    progress = tqdm(input_image, desc=f"vocab size [{vocab_len}]")

    print('Start training MBPE...')
    for image in progress:
        mbpe_tokenizer.train(image, dim=config.dim, min_freq=config.min_freq, root_min_freq=config.root_min_freq)
        encoded = mbpe_tokenizer.encode(image, dim=config.dim)
        outputs.append(encoded)
        vocab_len = mbpe_tokenizer.get_vocab_len()
        progress.set_description(f"vocab size [{vocab_len}]")

    if verbose:
        print('vocabulary:', mbpe_tokenizer.get_vocab())
        print('output_ids: ', outputs)
    
    return outputs

def train():
    config = Config()

    train_loader = load_mnist_dataset(config)

    mbpe_tokenizer = tokenizer.Tokenizer()
    outputs = train_tokenizer(train_loader, mbpe_tokenizer, config, verbose=False)
    output_ids, attention_mask = pad_sequences(outputs, max_length=config.max_length)
    labels = torch.roll(output_ids.clone(), shifts=-1, dims=1)
    labels[:, -1] = 0

    dataset = Dataset.from_dict({'input_ids': output_ids, 'attention_mask': attention_mask, 'labels': labels})
    print(dataset)
    
    model = get_model(config, vocab_size=mbpe_tokenizer.get_vocab_len())

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

if __name__ == "__main__":
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    train()