import gym
import numpy as np
import ant_hrl_maze
from PIL import Image

env = gym.make("AntWaypointHierarchical-v4")
env.reset()
while True:
    x, y = np.random.random(2) * 2 + 1
    env.env.reset_model(x, y)
    env.step(1)
    while True:
        try:
            env.render()
        except KeyboardInterrupt:
            break
