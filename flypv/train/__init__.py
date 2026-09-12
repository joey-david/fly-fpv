"""Training entry points.

The canonical trainer carries connectome state across control steps and uses
truncated BPTT.  ``flypv.train.ppo`` remains available for the original
stateless implementation and for reproducibility of old runs.
"""
from .recurrent_ppo import PPO, TrainConfig, train

__all__ = ["PPO", "TrainConfig", "train"]
