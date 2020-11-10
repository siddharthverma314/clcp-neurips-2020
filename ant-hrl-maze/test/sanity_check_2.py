import gym
import ant_hrl_maze
import random
import matplotlib.pyplot as plt

env = gym.make("AntMaze-v4")

actions = [random.randint(0, 19) for i in range(10)]

for _ in range(10):
    obs = [env.reset()]
    for a in actions:
        obs.append(env.step(a)[0])
    plt.plot([o[0] for o in obs], [o[1] for o in obs])
plt.show()
