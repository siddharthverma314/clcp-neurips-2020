from pyrl.sampler import BaseSampler
from pyrl.actor import GaussianActor
from pyrl.wrappers import TorchWrapper
import gym


def test_single_data_collection_integration():
    env = gym.make("Pendulum-v0")
    env = TorchWrapper(env, "cpu")
    policy = GaussianActor(env.observation_space, env.action_space, [256, 256])
    dc = BaseSampler(policy.action, 50, env)

    print(dc.log_hyperparams())
    print(dc.sample())
    print(dc.log_epoch())
