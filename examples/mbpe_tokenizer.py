import torch
from mbpe import basic
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


def tokenize(input, config, verbose=False):
    tokenizer = basic.Tokenizer()

    outputs = []

    vocab_len = tokenizer.get_vocab_len()
    progress = tqdm(input, desc=f"vocab size [{vocab_len}]")

    print('Start training MBPE...')
    for image in progress:
        tokenizer.train(image, dim=config.dim, min_freq=config.min_freq, root_min_freq=config.root_min_freq)
        tokenized = tokenizer.encode(image, dim=config.dim)
        outputs.append(tokenized)
        vocab_len = tokenizer.get_vocab_len()
        progress.set_description(f"vocab size [{vocab_len}]")

    if verbose:
        print('vocabulary:', tokenizer.get_vocab())
        print('output_ids: ', outputs)

    outputs, attention_mask = pad_sequences(outputs, max_length=config.max_length)
    return {'input_ids': outputs.tolist(), 'attention_mask': attention_mask.tolist()}