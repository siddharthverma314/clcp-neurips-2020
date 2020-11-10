import ant_hrl_maze
import numpy as np
import gym


def test_ant_reset_free():
    env = gym.make("AntResetFree-v4")
    obs1 = env.reset()
    assert isinstance(obs1, np.ndarray)
    obs2, _, _, _ = env.step(env.action_space.sample())
    assert isinstance(obs2, np.ndarray)
    assert not all(obs2 == obs1)
    assert all(obs2 == env.reset())
