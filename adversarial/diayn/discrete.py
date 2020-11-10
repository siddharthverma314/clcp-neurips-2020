from typing import List, Optional

import torch
import torch.distributions as pyd
from torch.functional import F
from contextlib import contextmanager
import gym

from pyrl.utils import MLP
from .base import BaseDiayn
from pyrl.logger import simpleloggable


@simpleloggable
class DiscreteDiayn(BaseDiayn):
    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        hidden_dim: List[int],
        _num_skills: int,
        _reward_weight: float,
        _lr: float = 1e-3,
        _device: str = "cpu",
        _truncate: Optional[int] = None,
    ) -> None:

        super().__init__(_truncate)
        self.num_skills = _num_skills
        self.reward_weight = _reward_weight
        self.device = _device
        self.z = None

        input_dim = self._input_shape(obs_space)
        self.model = MLP(input_dim, hidden_dim, self.num_skills).to(_device)
        self.optim = torch.optim.Adam(self.model.parameters(), _lr)

    def sample(self, batch_size: int) -> torch.Tensor:
        z = (
            self.z.repeat(batch_size)
            if isinstance(self.z, torch.Tensor)
            else torch.randint(self.num_skills, (batch_size,))
        )
        return F.one_hot(z, num_classes=self.num_skills,).float().to(self.device)

    @contextmanager
    def with_z(self, z):
        prev_z = self.z
        if isinstance(z, torch.Tensor):
            self.z = z
        else:
            self.z = torch.tensor(z)
        yield
        self.z = prev_z

    def train(self, obs: dict) -> None:
        obs, z = self._split_obs(obs)
        z = torch.argmax(z, dim=1, keepdim=True).squeeze(1)
        loss = F.cross_entropy(self.model.forward(obs), z)
        self.log("loss", loss)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def calc_rewards(self, obs: dict, eps: float = 1e-7) -> torch.Tensor:
        with torch.no_grad():
            obs, z = self._split_obs(obs)
            z = torch.argmax(z, dim=1, keepdim=True).squeeze(1)
            logits = self.model.forward(obs)
            rewards = (
                (self.reward_weight * -F.cross_entropy(logits, z, reduction="none"))
                .unsqueeze(1)
                .detach()
            )
            pred_z = pyd.Categorical(logits=logits).sample()
            self.log("accuracy", (pred_z == z).float().mean())
            self.log("rewards", rewards)
            return rewards
