from typing import Tuple
import gym
import numpy as np


class ResetFreeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._obs = None

    def _reset_if_needed(self):
        if self._obs is None:
            self._obs = self.env.reset()

    def reset(self) -> np.ndarray:
        self._reset_if_needed()
        return self._obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        nobs, reward, done, info = self.env.step(action)
        if self._obs is not None and not done:
            self._obs = nobs
        else:
            self._obs = None
        return nobs, reward, done, info
