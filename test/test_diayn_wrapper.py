from adversarial.diayn.discrete import DiscreteDiayn
from adversarial.diayn.continuous import ContinuousDiayn
from adversarial.env import DiaynWrapper, make_env
from pyrl.wrappers import DictWrapper, TorchWrapper
import torch
import gym


def abstract_test_diayn(diayn_class, device):
    env = make_env("InvertedPendulum-v2", device)
    diayn = diayn_class(env.observation_space, [256, 256], 1000, 2.0, _device=device)
    env = DiaynWrapper(env, diayn)

    # test device
    obs = env.reset()
    assert obs["diayn"].device == obs["observations"].device

    # test reset
    obs = env.reset()
    obs2 = env.reset()
    assert not torch.all(obs["diayn"] == obs2["diayn"])

    # test same rollout
    obs = env.reset()
    for _ in range(1000):
        action = torch.tensor(env.action_space.sample()).unsqueeze(0)
        obs2, _, _, _ = env.step(action)
        assert torch.all(obs["diayn"] == obs2["diayn"])


def test_discrete_diayn_wrapper():
    for device in "cpu", "cuda":
        abstract_test_diayn(DiscreteDiayn, device)


def test_continuous_diayn_wrapper():
    for device in "cpu", "cuda":
        abstract_test_diayn(ContinuousDiayn, device)
