from dataclasses import dataclass

@dataclass
class TrainConfig:
    rounds: int = 50
    client_frac: float = 0.1
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42
    noise_scale: float = 0.0  # Laplace b
    use_gpu: bool = False
