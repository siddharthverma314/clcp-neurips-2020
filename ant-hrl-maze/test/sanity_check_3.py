import ant_hrl_maze
import gym

env = gym.make("Ant-v4")
env.reset()
while True:
    _, _, done, _ = env.step(env.action_space.sample())
    print(done)
    env.render()
