from typing import Dict, Tuple

import abc
import gym
import torch
import numpy as np


class BaseDiayn(abc.ABC):
    def __init__(self, truncate):
        self.truncate = truncate

    @abc.abstractmethod
    def sample(batch_size: int) -> torch.Tensor:
        "Sample batch_size number of samples of z"

    @abc.abstractmethod
    def train(obs: Dict[str, torch.Tensor]):
        "Train z values from observations"

    @abc.abstractmethod
    def calc_rewards(obs: Dict[str, torch.Tensor]):
        "Predict the z values for a given observation"

    @property
    def observation_space(self):
        z = len(self.sample(1).squeeze())
        return gym.spaces.Box(
            low=np.repeat(np.float32("-inf"), z), high=np.repeat(np.float32("inf"), z)
        )

    def _input_shape(self, obs_space: gym.spaces.Dict):
        if self.truncate:
            return self.truncate
        return sum([len(v.low) for k, v in obs_space.spaces.items() if k != "diayn"])

    def _split_obs(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_obs = torch.cat([v for k, v in obs.items() if k != "diayn"], axis=1)
        diayn = obs["diayn"]
        if self.truncate:
            new_obs = new_obs[:, : self.truncate]
        return new_obs, diayn
