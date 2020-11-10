from typing import Tuple, Union, Callable

import torch
import gym
from contextlib import contextmanager
from pyrl.utils import collate, torchify
from pyrl.logger import Loggable


class BaseSampler(Loggable):
    def __init__(
        self, action_from_obs: Callable, _path_length: int, env: gym.Env,
    ) -> None:
        self.policy = action_from_obs
        self.path_length = _path_length
        self.env = env

        # for logging
        self._total_rewards = []
        self._infos = []

    @contextmanager
    def with_env(self, env: gym.Env):
        prev_env = self.env
        self.env = env
        yield
        self.env = prev_env

    def sample(
        self, start_obs: Union[None, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        data = []
        obs = start_obs if start_obs else self.env.reset()

        for _ in range(self.path_length):
            action = self.policy(obs)
            next_obs, reward, done, info = self.env.step(action)
            single_data = {
                "obs": obs,
                "act": action,
                "next_obs": next_obs,
                "done": done,
                "rew": reward,
            }
            data.append(single_data)
            obs = next_obs
            if done:
                break

        data = collate(data)

        self._total_rewards.append(data["rew"].sum())
        self._infos.append(torchify(info))
        return obs, data

    def log_local_hyperparams(self):
        return {"path_length": self.path_length}

    def log_local_epoch(self):
        epoch = {
            "total_reward": torch.stack(self._total_rewards),
            "info": collate(self._infos),
        }
        self._total_rewards = []
        self._infos = []
        return epoch
