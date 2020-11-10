import pickle
from pathlib import Path
import ant_hrl_maze
import numpy as np
from ant_hrl_maze.policy import Policy
import gym
import matplotlib.pyplot as plt


def load_policy(filepath) -> Policy:
    with open(filepath, "rb") as f:
        policy = pickle.load(f)
    return policy


def load_reset_free_policy():
    path = (
        Path(__file__).absolute().parent.parent
        / "policies"
        / "ours_reset_free"
        / "policy.pkl"
    )
    return load_policy(path)


def make_path(
    x: float, y: float, z: np.ndarray, env: gym.Env, policy: Policy, step=100,
) -> np.ndarray:
    offset = np.zeros_like(env.reset())
    offset[0] = x
    offset[1] = y

    obs = env.reset_model(x, y)
    obss = [obs]
    for i in range(step):
        obs, _, _, _ = env.step(policy.forward(obs - offset, z))
        obss.append(obs)

    obs = np.array(obss)
    offset = np.zeros_like(obs)
    offset[:, 0] = x
    offset[:, 1] = y
    obs -= offset
    return obs


def test_offset():
    policy = load_reset_free_policy()
    env = gym.make("Ant-v4")

    for skill in range(10):
        print(skill)
        z = np.zeros(10)
        z[skill] = 1
        paths = []
        print("SKILL", skill)
        for _ in range(10):
            x = np.random.random() * 100
            y = np.random.random() * 100
            print("XY", x, y)
            path = make_path(x, y, z, env, policy)
            paths.append(path)
        for path in paths:
            plt.plot(path[:, 0], path[:, 1])
        plt.show()
        # assert ((paths[1] - paths[0])**2).mean() < 1e-3


def test_robustness():
    policy = load_reset_free_policy()
    env = gym.make("Ant-v4")

    for skill in range(10):
        z = np.zeros(10)
        z[skill] = 1

        for _ in range(5):
            # reset
            qpos = env.init_qpos.copy()
            qvel = env.init_qvel.copy()
            qpos += np.random.randn(*qpos.shape) * 1e-1
            qpos[:2] = 0
            # qvel += np.random.randn(*qvel.shape) * 1e-2
            env.set_state(qpos, qvel)
            obs = np.concatenate((qpos, qvel))

            obss = [obs]

            for _ in range(200):
                obs, _, _, _ = env.step(policy.forward(obs, z))
                obss.append(obs)

            obss = np.array(obss)
            plt.plot(obss[:, 0], obss[:, 1])
            plt.plot(obss[-1, 0], obss[-1, 1], "r.")
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)

        plt.show()


def test_compositional_robustness():
    policy = load_reset_free_policy()
    env = gym.make("Ant-v4")

    for skill in range(10):
        z = np.zeros(10)
        z[skill] = 1

        obs = env.reset()
        for _ in range(5):
            obss = [obs]
            for _ in range(200):
                obs, _, _, _ = env.step(policy.forward(obs, z))
                obss.append(obs)
            obss = np.array(obss)

            plt.plot(obss[:, 0], obss[:, 1])
            plt.plot(obss[-1, 0], obss[-1, 1], "r.")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.show()


def test_chelsea_idea():
    policy = load_reset_free_policy()
    env = gym.make("Ant-v4")

    for skill in range(10):
        z = np.zeros(10, dtype=np.float32)
        z[skill] = 1

        obs = env.reset()
        print(obs)
        obss = [obs]
        for _ in range(50):
            offset = np.zeros_like(obs)
            offset[:2] = obs[:2]
            for _ in range(20):
                action = policy.forward(obs - offset, z)
                obs, _, _, _ = env.step(action)
                obss.append(obs)
            # for _ in range(2):
            #    env.step(env.action_space.sample())
        obss = np.array([obss[0], obss[-1]])
        plt.plot(obss[:, 0], obss[:, 1])
        plt.plot(obss[-1, 0], obss[-1, 1])
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()


# def test_chelsea_idea():
#     policy = load_reset_free_policy()
#     env = gym.make('Ant-v4')
#
#     for skill in range(10):
#         z = np.zeros(10, dtype=np.float32)
#         z[skill] = 1.
#
#         obs = env.reset()
#         obss = [obs]
#         for i in range(1000):
#             if i % 20 == 0:
#                 offset = np.zeros_like(obs)
#                 offset[:2] = obs[:2]
#
#             cur_action = policy.forward(obs - offset, z)
#             print("OBSERVATION", obs - offset)
#             print("ACTION", cur_action)
#             obs, _, done, info = env.step(cur_action)
#             obss.append(obs)
#         obss = np.array([obss[0], obss[-1]])
#         plt.plot(obss[:, 0], obss[:, 1])
#         plt.plot(obss[-1, 0], obss[-1, 1])
#     plt.xlim(-5, 5)
#     plt.ylim(-5, 5)
#     plt.show()
