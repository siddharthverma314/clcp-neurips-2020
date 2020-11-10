from ant_hrl_maze.ant_v4 import AntEnv
from .hierarchical_wrapper import HierarchicalWrapper
from pathlib import Path
import numpy as np
import gym


class AntWaypointEnv(AntEnv):
    GOAL_BONUS = 1
    THRESHOLD = 0.5

    def __init__(
        self, method="ours_reset_free", *args, **kwargs,
    ):
        self.goals = np.array([(0, 2), (2, 2), (2, 0)])
        self.num_goals = 3
        self.current_goal = 0

        # add site
        xml_path = Path(__file__).absolute().parent / "assets" / "ant_site.xml"
        super().__init__(str(xml_path), *args, **kwargs)

    def _get_xy(self):
        return super()._get_obs()[:2].copy()

    def _get_obs(self):
        return self._get_xy()

    def _pop_goal(self):
        print("GOAL")
        self.current_goal += 1

    def _current_goal(self):
        return self.goals[self.current_goal]

    def _finished(self):
        return self.current_goal >= self.num_goals

    def reset(self):
        self.current_goal = 0
        return super().reset()

    def step(self, action):
        _, _, done, _ = super().step(action)

        d = 0
        if not self._finished():
            d = np.linalg.norm(self._get_xy() - self._current_goal())
            site_id = self.sim.model.site_name2id("goal")
            xyz = self.sim.model.site_pos[site_id]
            xyz[:2] = self._current_goal()[:2]

        reward = np.exp(-(d ** 2)) + self.current_goal

        if d < self.THRESHOLD:
            reward += self.GOAL_BONUS
            self._pop_goal()

        info = {"num_goals_remaining": self.num_goals - self.current_goal}

        return self._get_obs(), reward, self._finished(), info


def hierarchical_waypoint(ant_waypoint_args={}, wrapper_args={},) -> gym.Env:
    ant_waypoint_env = AntWaypointEnv(**ant_waypoint_args)
    wrapped_env = HierarchicalWrapper(ant_waypoint_env, **wrapper_args)
    return wrapped_env
