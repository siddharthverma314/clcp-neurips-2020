import gym
import torch
from pyrl.wrappers.torch import TorchWrapper


def test_torch_wrapper():
    for device in "cpu", "cuda":
        env = gym.make("Pendulum-v0")
        env = TorchWrapper(env, device=device)

        def foo(obj):
            assert obj.dim() == 2
            assert obj.device.type == device

        obs = env.reset()
        foo(obs)

        act = torch.tensor(env.action_space.sample()).unsqueeze(0).to(device)
        foo(act)

        obs, reward, done, info = env.step(act)
        foo(obs)
        foo(reward)
        foo(done)
        assert isinstance(info, dict)
