import gym
import ant_hrl_maze
import matplotlib.pyplot as plt
import numpy as np


def test_each_action():
    env = gym.make("AntMaze-v4")
    for skill in range(10):
        obs = env.reset()
        nobs, _, _, _ = env.step(skill)
        obs = np.array([obs, nobs])
        plt.plot(obs[:, 0], obs[:, 1])
    plt.show()


if __name__ == "__main__":
    test_each_action()
