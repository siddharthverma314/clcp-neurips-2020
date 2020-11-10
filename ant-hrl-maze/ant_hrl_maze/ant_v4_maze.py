from typing import Tuple
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import itertools
import os
from copy import deepcopy
from .ant_v4 import AntEnv
from .policy import Policy
import gym
import pickle
from os.path import join, dirname

RESET = R = 2  # Reset position.
GOAL = G = 3

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
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, G, 1],
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
    [1, R, 0, 0, 1, 0, G, 1],
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


class AntMazeEnv(AntEnv):
    def __init__(
        self,
        policy_steps=1,
        maze_size_scaling=5,
        maze_map="u",
        maze_height=0.5,
        reward_type="dense",
        num_skills=10,
        method="ours_reset_free",
        use_policy=False,
        meta_chelsea_hack=50,
        **ant_kwargs
    ) -> None:

        if maze_map == "u":
            maze_map = U_MAZE
        elif maze_map == "big":
            maze_map = BIG_MAZE
        elif maze_map == "hardest":
            maze_map = HARDEST_MAZE

        # create xml
        self._maze_map = np.array(maze_map)
        self._maze_height = maze_height
        self._maze_size_scaling = maze_size_scaling
        self._meta_chelsea_hack = meta_chelsea_hack

        xml_path = os.path.join(os.path.dirname(__file__), "assets", "ant.xml")
        tree = ET.parse(xml_path)
        self._worldbody = tree.find(".//worldbody")

        for i, j in itertools.product(*map(range, self._maze_map.shape)):
            elem = self._maze_map[i, j]
            if elem == 1:
                self._add_block(i, j)
            elif elem == R:
                self._reset_pos = self._rowcol_to_xy(i + 0.25, j + 0.25)
            elif elem == G:
                self._goal_pos = self._rowcol_to_xy(i + 0.25, j + 0.25)

        _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
        tree.write(file_path)

        # create policy
        self._use_policy = use_policy
        if use_policy:
            pkl_path = join(
                dirname(dirname(__file__)), "policies", method, "policy.pkl"
            )
            self._policy: Policy = pickle.load(open(pkl_path, "rb"))
        else:
            pkl_path = join(
                dirname(dirname(__file__)), "policies", method, "actions.pkl"
            )
            self._actions: np.ndarray = pickle.load(open(pkl_path, "rb"))

        self._policy_steps = policy_steps
        self.num_skills = num_skills

        if not self._reset_pos or not self._goal_pos:
            raise AssertionError("Robot or goal not found!")

        self.is_initialized = False
        super().__init__(
            xml_file=file_path, **ant_kwargs,
        )

        # set spaces
        self.is_initialized = True
        self.action_space = gym.spaces.Discrete(self.num_skills)

    def _add_block(self, i, j):
        ET.SubElement(
            self._worldbody,
            "geom",
            name="block_%d_%d" % (i, j),
            pos="%f %f %f"
            % (
                *self._rowcol_to_xy(i, j),
                self._maze_height / 2 * self._maze_size_scaling,
            ),
            size="%f %f %f"
            % (
                0.5 * self._maze_size_scaling,
                0.5 * self._maze_size_scaling,
                self._maze_height / 2 * self._maze_size_scaling,
            ),
            type="box",
            material="",
            contype="1",
            conaffinity="1",
            rgba="0.7 0.5 0.3 1.0",
        )

    def _rowcol_to_xy(self, row, col):
        x = col * self._maze_size_scaling
        y = row * self._maze_size_scaling
        return (x, y)

    def reset_model(self):
        super().reset_model(*self._reset_pos)
        return self._get_obs()

    def step(self, action: int, render=False) -> Tuple[dict, float, bool, dict]:
        action = np.squeeze(action)
        if not self.is_initialized:
            action = 0

        fall = True
        z = np.zeros(self.num_skills, dtype=np.float32)
        z[action] = 1.0

        self.images = []
        for i in range(self._policy_steps):
            offset = np.zeros_like(self._get_obs())
            offset[:2] = self._get_obs()[:2]

            for i in range(self._meta_chelsea_hack):
                if self._use_policy:
                    cur_action = self._policy.forward(self._get_obs() - offset, z)
                else:
                    cur_action = self._actions[action, i]

                obs, _, done, info = super().step(cur_action)

                if render:
                    self.images.append(self.render("rgb_array"))

                # check terminate condition
                d = np.linalg.norm(self._get_obs()[:2] - self._goal_pos)
                if d < 2:
                    print("============")
                    print("----YAY-----")
                    print("============")
                    fall = False
                    done = True
                if done:
                    break

        bonus = -1000 if fall else 1000
        reward = -d + 1000 * np.exp(-(d ** 2) / 2)
        return self._get_obs(), reward if not done else bonus, done, info

    def viewer_setup(self):
        midpoint = self._rowcol_to_xy(*[(i - 1) / 2 for i in self._maze_map.shape])
        from mujoco_py.generated import const

        self.viewer.cam.type = const.CAMERA_FREE
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[:2] = midpoint
        self.viewer.cam.lookat[2] = 0
        self.viewer.cam.elevation = -65
        self.viewer.cam.distance = 50
        self.viewer.cam.azimuth = 45
