import torch
from typing import Tuple
from gym import Env, Wrapper
from torch import Tensor
from pyrl.utils import torchify, untorchify


class TorchWrapper(Wrapper):
    def __init__(self, env: Env, device: str) -> None:
        super().__init__(env)
        self.device = torch.device(device)

    def reset(self) -> dict:
        obs = self.env.reset()
        return torchify(obs, self.device)

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, dict]:
        action = untorchify(action)
        next_obs, reward, done, info = self.env.step(action)
        return (
            torchify(next_obs, self.device),
            torchify(reward, self.device),
            torchify(done, self.device),
            info,
        )

    def render(self, mode):
        if mode == "rgb_array":
            return torch.tensor(self.env.render("rgb_array").copy())
        return self.env.render(mode)
