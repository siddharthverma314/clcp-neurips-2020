import cpprb
import torch
import numpy as np
from gym import Space
from pyrl.logger import simpleloggable
from pyrl.utils import untorchify, torchify
from pyrl.transforms import Flatten, Unflatten


@simpleloggable
class ReplayBuffer:
    def __init__(
        self,
        obs_spec: Space,
        act_spec: Space,
        _capacity: int,
        _batch_size: int,
        _device: str,
    ):
        self.obs_flat = Flatten(obs_spec)
        self.obs_unflat = Unflatten(obs_spec)
        self.act_flat = Flatten(act_spec)
        self.act_unflat = Unflatten(act_spec)

        spec = {
            "obs": {"dtype": np.float32, "shape": self.obs_flat.dim},
            "act": {"dtype": np.float32, "shape": self.act_flat.dim},
            "next_obs": {"dtype": np.float32, "shape": self.obs_flat.dim},
            "rew": {"dtype": np.float32, "shape": 1},
            "done": {"dtype": np.float32, "shape": 1},
        }

        self.buffer: cpprb.ReplayBuffer = cpprb.create_buffer(_capacity, spec)
        self.batch_size = _batch_size
        self.device = torch.device(_device)

    def __len__(self):
        return len(self.buffer)

    def add(self, batch):
        self.buffer.add(
            **untorchify(
                {
                    "obs": self.obs_flat(batch["obs"]),
                    "act": self.act_flat(batch["act"]),
                    "next_obs": self.obs_flat(batch["next_obs"]),
                    "rew": batch["rew"],
                    "done": batch["done"],
                }
            )
        )

    def sample(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size

        batch = torchify(self.buffer.sample(batch_size), self.device)
        return {
            "obs": self.obs_unflat(batch["obs"]),
            "act": self.act_unflat(batch["act"]),
            "next_obs": self.obs_unflat(batch["next_obs"]),
            "rew": batch["rew"],
            "done": batch["done"],
        }
