import gym
from ant_hrl_maze.reset_free_wrapper import ResetFreeWrapper


class FakeEnv(gym.Env):
    "Basically counting"

    def __init__(self):
        pass

    def reset(self):
        self.num = 0
        return self.num

    def step(self, _):
        self.num += 1
        if self.num == 10:
            done = True
        else:
            done = False
        return self.num, 0, done, {}


def test_reset():
    env = ResetFreeWrapper(FakeEnv())
    assert env.reset() == 0
    env.step(None)
    assert env.reset() == 1
    env.step(None)
    assert env.reset() == 2


def test_reset_done():
    env = ResetFreeWrapper(FakeEnv())
    assert env.reset() == 0
    for _ in range(10):
        env.step(None)
    assert env.reset() == 0
    for _ in range(20):
        env.step(None)
    print(env._obs)
    assert env.reset() == 0
