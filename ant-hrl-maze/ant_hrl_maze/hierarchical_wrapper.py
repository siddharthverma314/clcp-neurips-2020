from typing import Tuple
import gym
from ant_hrl_maze.policy import Policy
import pickle
from pathlib import Path
from ant_hrl_maze.ant_v4 import AntEnv
import numpy as np


class HierarchicalWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        use_policy=True,
        backward_steps=0,
        policy_steps=100,
        num_skills=10,
        normalize=True,
        method="ours_reset_free",
    ):
        super().__init__(env)

        pkl_path = Path(__file__).absolute().parent.parent / "policies" / method
        if use_policy:
            pkl_path /= "policy.pkl"
        else:
            pkl_path /= "actions.pkl"

        self._use_policy = use_policy
        with open(pkl_path, "rb") as f:
            if use_policy:
                self.policy: Policy = pickle.load(f)
            else:
                self.actions: np.ndarray = pickle.load(f)

        self._policy_steps = policy_steps
        self._backward_steps = backward_steps
        self.num_skills = num_skills
        self.normalize = normalize

        self.action_space = gym.spaces.Discrete(self.num_skills)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action: int, render=False) -> Tuple[dict, float, bool, dict]:
        action = np.squeeze(action)
        z = np.zeros(self.num_skills, dtype=np.float32)
        z[action] = 1.0

        obs = AntEnv._get_obs(self.env)
        offset = np.zeros_like(obs)
        if self.normalize:
            offset[:2] = obs[:2].copy()

        total_reward = 0

        done = False
        if self._use_policy:
            for i in range(self._backward_steps):
                cur_action = self.policy.backward(AntEnv._get_obs(self.env) - offset)
                obs, reward, done, info = self.env.step(cur_action)
                total_reward += reward
                if done:
                    break

        obs = AntEnv._get_obs(self.env)
        offset = np.zeros_like(obs)
        offset[:2] = obs[:2]

        for i in range(self._policy_steps):
            if done:
                break

            if self._use_policy:
                cur_action = self.policy.forward(AntEnv._get_obs(self.env) - offset, z)
            else:
                cur_action = self.actions[action, i]

            obs, reward, done, info = self.env.step(cur_action)
            total_reward += reward

        return self.env._get_obs(), total_reward, done, info
