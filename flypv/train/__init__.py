"""Training entry points."""
from .config import TrainConfig
from .curriculum import PPO, train

__all__ = ["PPO", "TrainConfig", "train"]
