import ant_hrl_maze
import gym
import numpy as np


env = gym.make("AntMaze-v4")
env.reset()
while True:
    try:
        env.render()
    except KeyboardInterrupt:
        break

env.render()
while True:
    action = int(input("Action:"))
    obs, reward, done, info = env.step(action, render=True)
    print(obs, reward, done, info)
    env.render()
    if done:
        env.reset()
    env.render()
