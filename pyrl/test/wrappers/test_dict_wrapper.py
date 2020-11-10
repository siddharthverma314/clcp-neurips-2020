import gym
import torch
from pyrl.wrappers import DictWrapper, TorchWrapper


def test_wrapper_env():
    env = gym.make("Pendulum-v0")
    env = DictWrapper(env)
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert list(env.reset().keys()) == ["observations"]

    obs, a, b, c = env.step(env.action_space.sample())
    assert isinstance(obs, dict) and list(obs.keys()) == ["observations"]
    assert not isinstance(a, dict)
    assert not isinstance(b, dict)


def test_double_wrap_env():
    env = gym.make("Pendulum-v0")
    env = DictWrapper(env)
    env = DictWrapper(env)

    obs_space = env.observation_space.spaces["observations"]
    assert not isinstance(obs_space, gym.spaces.Dict) or isinstance(obs_space, dict)
    assert not isinstance(env.reset()["observations"], dict)


def test_composition_torch_wrapper_dict_wrapper():
    for device in "cpu", "cuda":
        env = gym.make("Pendulum-v0")
        env = TorchWrapper(env, device)
        env = DictWrapper(env)

        obs = env.reset()["observations"]
        assert isinstance(obs, torch.Tensor) and obs.dim() == 2
