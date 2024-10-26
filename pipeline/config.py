from dataclasses import dataclass

@dataclass
class Config:
    # Tokenizer configurations
    dim: tuple = (2, 2)
    min_freq: int = 2
    root_min_freq: int = 2
    size = 1000        # number of images to train on; 0 stands for full dataset
    max_length: None = None

    # Model configurations
    model_name: str = "distilbert/distilgpt2"
    
    # Training configurations
    batch_size: int = 32
    num_train_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01