from typing import Tuple
import numpy as np
import gym

RESET = R = 2  # Reset position.
GOAL = G = 3  # Goal position

# Maze specifications for dataset generation
STRAIGHT_MAZE = [[1, 1, 1, 1, 1], [1, R, 0, G, 1], [1, 1, 1, 1, 1]]

U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

BIG_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, G, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, G, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, G, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

HARDEST_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 1, G, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, G, 0, 1, 0, 0, G, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, G, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, G, 0, G, 1, 0, G, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Maze specifications for evaluation
U_MAZE_TEST = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

BIG_MAZE_TEST = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

HARDEST_MAZE_TEST = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, G, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


class TestEnv(gym.Env):
    def __init__(self, maze=U_MAZE) -> None:
        # set spaces
        tmp = np.repeat(float("inf"), 2)
        self.observation_space = gym.spaces.Box(low=-tmp, high=tmp)
        self.action_space = gym.spaces.Discrete(4)

        # initialize maze
        self.maze = np.array(maze)
        self.goal_pos = np.concatenate(np.where(self.maze == G))
        self.reset_pos = np.concatenate(np.where(self.maze == R))

        # reset
        self.reset()

    def reset(self):
        self.pos = self.reset_pos + np.array([0.5, 0.5])
        return self.pos

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        direction = (
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, -1]),
            np.array([0, 1]),
        )[action]

        # check if next is in a block
        next_pos = self.pos + direction
        next_index = self.maze[int(next_pos[0]), int(next_pos[1])]
        if next_index != 1:
            self.pos = next_pos

        # calculate reward
        d = np.linalg.norm(self.pos - self.goal_pos + np.array([0.5, 0.5]))
        done = True if next_index == G else False
        reward = 10 * np.exp(-(d ** 2) / 3)
        return self.pos, reward, done, {}
