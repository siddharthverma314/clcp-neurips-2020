import gym
import ant_hrl_maze

env = gym.make("AntMaze-v4")
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
    elif inp.isnumeric():
        print(env.step(int(inp), render=True)[0])
    env.render()
