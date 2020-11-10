import gym
import ant_hrl_maze
import numpy as np


def test_ant_waypoint():
    env = gym.make("AntWaypoint-v4")
    assert len(env.goals) == 5
    goals = np.array(env.goals)
    assert np.min(goals) > -10 and np.max(goals) < 10
    env._pop_goal()
    assert env.current_goal == 1
    env.reset()
    assert env.current_goal == 0


def test_ant_waypoint_teleport():
    env = gym.make("AntWaypoint-v4")
    assert len(env.goals) == 5
    goals = np.array(env.goals)
    action = np.zeros(8)
    for i in range(len(goals)):
        env.reset_model(*goals[i])
        _, reward, done, _ = env.step(action)
        print("I", i)
        assert reward >= env.GOAL_BONUS
    assert done
