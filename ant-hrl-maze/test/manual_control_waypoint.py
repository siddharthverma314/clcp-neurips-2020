import gym
import ant_hrl_maze
from PIL import Image

env = gym.make("AntWaypointHierarchical-v4")
env.reset()
while True:
    try:
        env.render()
    except KeyboardInterrupt:
        break

while True:
    inp = input("Action")
    if inp == "r":
        env.reset()
    elif inp == "c":
        env.render("rgb_array")
    elif inp.isnumeric():
        print(env.step(int(inp), render=True)[0])
    env.render()
