from typing import Union, Tuple
from gym import spaces, Wrapper
from torch import Tensor
from pyrl.wrappers import TorchWrapper
from adversarial.diayn.base import BaseDiayn


class DiaynWrapper(Wrapper):
    def __init__(self, env: TorchWrapper, diayn: BaseDiayn):
        assert isinstance(env.observation_space, spaces.Dict)
        super().__init__(env)
        self.diayn = diayn
        self.observation_space = spaces.Dict(
            {**env.observation_space.spaces, "diayn": self.diayn.observation_space}
        )

    def _process_obs(self, obs: dict) -> dict:
        return {**obs, "diayn": self.z}

    def reset(self) -> dict:
        self.z = self.diayn.sample(1)
        return self._process_obs(self.env.reset())

    def step(self, action: Tensor) -> Tuple[dict, Tensor, Tensor, dict]:
        next_obs, reward, done, info = self.env.step(action)
        return self._process_obs(next_obs), reward, done, info
