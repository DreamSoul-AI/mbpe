from transformers import AutoModelForCausalLM

def get_model(config, vocab_size=None):
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    if vocab_size is not None:
        model.resize_token_embeddings(vocab_size)
    return model