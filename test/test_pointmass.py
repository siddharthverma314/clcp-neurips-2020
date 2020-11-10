import adversarial.env
from adversarial.env import make_env
import gym
import torch


def test_pointmass():
    env = gym.make("PointMass-v0")
    env.reset()
    env.step(env.action_space.sample())


def test_pointmass_integration():
    for device in "cpu", "cuda":
        env = make_env("PointMass-v0", "cpu")
        env.reset()
        action = torch.tensor(env.action_space.sample()).to(device)
        env.step(action)
