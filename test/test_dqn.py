from adversarial.actor import DqnActor
from adversarial.critic import DqnCritic
from adversarial.env import make_env
from pyrl.utils import torchify, collate
import torch


def test_critic():
    for device in "cpu", "cuda":
        env = make_env("CartPole-v1", device)
        critic = DqnCritic(env.observation_space, env.action_space, [256, 256], device)
        for _ in range(300):
            critic(
                torchify(env.observation_space.sample(), device),
                torchify(env.action_space.sample(), device),
            )


def test_actor():
    for device in "cpu", "cuda":
        env = make_env("CartPole-v1", device)
        critic = DqnCritic(env.observation_space, env.action_space, [256, 256], device)
        actor = DqnActor(critic, env.observation_space, env.action_space, device)
        for _ in range(300):
            obs = collate(
                [torchify(env.observation_space.sample(), device) for _ in range(20)]
            )
            act = actor(obs)
            rand_act = torch.cat(
                [torchify(env.action_space.sample(), device) for _ in range(20)]
            )
            assert torch.all(critic(obs, act) >= critic(obs, rand_act))
