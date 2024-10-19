from transformers import AutoModelForCausalLM

def get_model(config):
    return AutoModelForCausalLM.from_pretrained(config.model_name)