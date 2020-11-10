import gym
from .ant_v4 import AntEnv
from .reset_free_wrapper import ResetFreeWrapper

gym.register(
    id="Ant-v4", entry_point=AntEnv,
)

gym.register(
    id="AntResetFree-v4",
    entry_point=lambda *args, **kwargs: ResetFreeWrapper(AntEnv(*args, **kwargs)),
)

gym.register(
    id="AntRandom-v4",
    entry_point="ant_hrl_maze.ant_v4:AntEnv",
    kwargs={"reset_noise_scale": 3,},
)

gym.register(
    id="AntTest-v4", entry_point="ant_hrl_maze.ant_v4_maze_test:TestEnv",
)

gym.register(
    id="AntMaze-v4", entry_point="ant_hrl_maze.ant_v4_maze:AntMazeEnv",
)

gym.register(
    id="AntWaypoint-v4", entry_point="ant_hrl_maze.waypoint:AntWaypointEnv",
)


gym.register(
    id="AntWaypointHierarchical-v4",
    entry_point="ant_hrl_maze.waypoint:hierarchical_waypoint",
)
