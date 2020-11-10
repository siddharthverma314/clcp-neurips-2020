from typing import Union, Tuple
from gym import spaces, Wrapper
from torch import Tensor
from .torch import TorchWrapper


class DictWrapper(Wrapper):
    def __init__(self, env: TorchWrapper):
        super().__init__(env)

        if not isinstance(self.observation_space, spaces.Dict):
            self.observation_space = spaces.Dict(
                {"observations": self.observation_space}
            )

    @staticmethod
    def _process_obs(obs: Union[dict, Tensor]) -> dict:
        if not isinstance(obs, dict):
            obs = {"observations": obs}
        return obs

    def reset(self) -> dict:
        obs = self.env.reset()
        return self._process_obs(obs)

    def step(self, action: Tensor) -> Tuple[dict, Tensor, Tensor, dict]:
        next_obs, reward, done, info = self.env.step(action)
        return self._process_obs(next_obs), reward, done, info
