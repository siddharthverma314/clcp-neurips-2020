import ant_hrl_maze
import gym


def test_hierarchical_waypoint():
    env = gym.make("AntWaypointHierarchical-v4")
    env.reset()
    for _ in range(10):
        obs, _, _, _ = env.step(env.action_space.sample())
        assert len(obs.shape) == 1
