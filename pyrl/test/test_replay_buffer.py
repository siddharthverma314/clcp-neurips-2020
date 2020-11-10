from pyrl.utils import create_random_space, torchify
from pyrl.replay_buffer import ReplayBuffer
from flatten_dict import flatten
import torch


def test_integration():
    for device in "cpu", "cuda":
        obs_space = create_random_space()
        act_space = create_random_space()
        buf = ReplayBuffer(obs_space, act_space, int(1e5), 1, device)
        print(buf.log_hyperparams())
        print("OBSSPEC", obs_space)
        print("ACTSPEC", act_space)

        step = {
            "obs": torchify(obs_space.sample(), device),
            "act": torchify(act_space.sample(), device),
            "rew": torchify(1.0, device),
            "next_obs": torchify(obs_space.sample(), device),
            "done": torchify(0, device),
        }
        buf.add(step)

        step2 = buf.sample()
        step = flatten(step)
        step2 = flatten(step2)
        assert step.keys() == step2.keys()
        for k in step:
            assert torch.all(step[k].cpu() == step2[k].cpu())

        print(buf.log_epoch())
