from dataclasses import dataclass

@dataclass
class ModelArgs:
    vocab_size: int = 12000
    d: int = 8  # hidden dim per head
    h_q: int = 8  # number of query heads
    h_kv: int = 4  # number of key/value heads
    s: int = 512  # maximum sequence length
    n_layers: int = 5  # number of layers
    norm_eps: float = 1e-5
    dropout: float = 0.125

@dataclass
class TrainArgs:
    data_dir: str = "../data"
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 2e-2
    checkpoint_steps: int = 10
    max_steps: int = 30000