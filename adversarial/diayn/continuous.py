from typing import List, Optional

import torch
import torch.distributions as pyd
import gym
from torch.functional import F
from .base import BaseDiayn
from pyrl.utils import MLP
from pyrl.logger import simpleloggable


@simpleloggable
class ContinuousDiayn(BaseDiayn):
    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        hidden_dim: List[int],
        _num_skills: int,
        _reward_weight: float,
        _lr: float = 3e-4,
        _device: str = "cpu",
        _weight_clip: float = 50,
        _truncate: Optional[int] = None,
    ) -> None:

        super().__init__(_truncate)
        self.num_skills = _num_skills
        self.reward_weight = _reward_weight
        self.device = _device
        self.weight_clip = _weight_clip

        input_dim = self._input_shape(obs_space)
        self.model = MLP(input_dim, hidden_dim, self.num_skills * 2).to(_device)
        self.optim = torch.optim.Adam(self.model.parameters(), _lr)

    def sample(self, batch_size: int) -> torch.Tensor:
        return (
            pyd.Normal(0, 1)
            .sample((batch_size, self.num_skills))
            .float()
            .to(self.device)
        )

    def train(self, obs: dict) -> None:
        obs, z = self._split_obs(obs)
        mu, var = self.model.forward(obs).split(self.num_skills, 1)
        pred_z = pyd.Normal(mu, torch.exp(var)).rsample()
        loss = F.mse_loss(pred_z, z)
        self.log("loss", loss)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def calc_rewards(self, obs: dict, eps: float = 1e-7) -> torch.Tensor:
        with torch.no_grad():
            obs, z = self._split_obs(obs)
            mu, var = self.model.forward(obs).split(self.num_skills, 1)
            rewards = (
                (
                    (
                        pyd.Normal(mu, torch.exp(var)).log_prob(z).sum(1)
                        - pyd.Normal(0, 1).log_prob(z).sum(1)
                    )
                    * self.reward_weight
                )
                .clamp(-self.weight_clip, self.weight_clip)
                .unsqueeze(1)
                .detach()
            )
            self.log("rewards", rewards)
            return rewards
