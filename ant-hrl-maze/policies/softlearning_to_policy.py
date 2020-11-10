import gym
import numpy as np
import pickle
from softlearning.policies.utils import get_policy_from_variant
from softlearning.environments.utils import get_environment_from_params
from ant_hrl_maze.policy import SoftlearningPolicy
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def load_policy(path):
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    variant = checkpoint["variant"]
    env_params = variant["environment_params"]["training"]
    alice_params = variant["alice"]
    bob_params = variant["bob"]
    num_skills = alice_params["algorithm_params"]["discriminator_params"]["num_skills"]

    # bob policy
    env = get_environment_from_params(env_params)
    bob_policy = get_policy_from_variant(bob_params, env)
    bob_policy.set_weights(checkpoint["policy_weights"]["bob"])
    bob_policy._deterministic = True

    # alice policy
    env._observation_space.spaces["diayn"] = gym.spaces.Box(
        low=np.repeat(0, num_skills), high=np.repeat(1, num_skills),
    )
    env.observation_keys += ("diayn",)

    alice_policy = get_policy_from_variant(alice_params, env)
    alice_policy.set_weights(checkpoint["policy_weights"]["alice"])
    alice_policy._deterministic = True

    return env, alice_policy, bob_policy, num_skills


def generate_data(env, policy, skill, num_skills, step=200):
    obs = env.reset()

    observations = [obs["observations"]]
    actions = []

    z = np.zeros(num_skills)
    z[skill] = 1

    for _ in range(step):
        action = policy.forward(obs["observations"], z)
        actions.append(action)
        obs, _, _, _ = env.step(action)
        observations.append(obs["observations"])

    observations = np.array(observations)
    actions = np.array(actions)
    return observations, actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method_dir")
    args = parser.parse_args()

    path = Path(__file__).absolute().parent / args.method_dir
    env, alice_policy, bob_policy, num_skills = load_policy(path / "checkpoint.pkl")
    policy = SoftlearningPolicy(alice_policy, bob_policy)

    print("Dumping policy")
    with open(path / "policy.pkl", "wb") as f:
        pickle.dump(policy, f)

    print("Dumping observations and actions")
    observations = []
    actions = []
    for skill in range(num_skills):
        o, a = generate_data(env, policy, skill, num_skills)
        observations.append(o)
        actions.append(a)
    observations = np.array(observations)
    actions = np.array(actions)

    np.save(path / "observations.npy", observations)
    with open(path / "actions.pkl", "wb") as f:
        pickle.dump(actions, f)
    plt.cla()
    for i in range(len(observations)):
        plt.plot(observations[i, :, 0], observations[i, :, 1])
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.savefig(path / "plot.png")
