from adversarial.env import make_env, DiaynWrapper
from adversarial.diayn import DiscreteDiayn
from pyrl.actor import TanhGaussianActor
from pyrl.utils import torchify, untorchify
from ant_hrl_maze.policy import Policy
from pathlib import Path
import torch
import json


class AdversarialPolicy(Policy):
    @staticmethod
    def check(path: str):
        path = Path(path).absolute()
        if not (path.parent / "hyperparams.json").exists():
            return False

        with open(path.parent / "hyperparams.json", "r") as f:
            hyperparams = json.load(f)
        return "alice_diayn" in hyperparams

    def __init__(self, path: str, device: str):
        path = Path(path).absolute()
        with open(path.parent / "hyperparams.json", "r") as f:
            hyperparams = json.load(f)

        checkpoint = torch.load(path, "cpu")
        self.ns = hyperparams["alice_diayn"]["num_skills"]
        env = make_env("Ant-v4", device)

        # bob
        self.backward_policy = TanhGaussianActor(
            obs_spec=env.observation_space,
            act_spec=env.action_space,
            hidden_dim=hyperparams["bob"]["actor"]["policy"]["hidden_dim"],
        )
        self.backward_policy.load_state_dict(checkpoint["bob"]["actor"]["state_dict"])
        self.backward_policy.to(device)

        # alice
        diayn = DiscreteDiayn(
            env.observation_space,
            hyperparams["alice_diayn"]["model"]["hidden_dim"],
            self.ns,
            hyperparams["alice_diayn"]["reward_weight"],
            _truncate=hyperparams["alice_diayn"]["truncate"],
        )
        env = DiaynWrapper(env, diayn)
        self.forward_policy = TanhGaussianActor(
            obs_spec=env.observation_space,
            act_spec=env.action_space,
            hidden_dim=hyperparams["alice"]["actor"]["policy"]["hidden_dim"],
        )
        self.forward_policy.load_state_dict(checkpoint["alice"]["actor"]["state_dict"])
        self.forward_policy.to(device)
        self.device = device

    @property
    def num_skills(self):
        return self.ns

    def forward(self, obs, skill):
        obs = {
            "observations": torchify(obs, self.device),
            "diayn": torchify(skill, self.device),
        }
        act = self.forward_policy.action(obs, deterministic=True)
        return untorchify(act)

    def backward(self, obs):
        obs = {"observations": torchify(obs, self.device)}
        act = self.backward_policy.action(obs, deterministic=True)
        return untorchify(act)


class DiaynPolicy(Policy):
    @staticmethod
    def check(path: str):
        path = Path(path).absolute()
        if not (path.parent / "hyperparams.json").exists():
            return False
        with open(path.parent / "hyperparams.json", "r") as f:
            hyperparams = json.load(f)
        return "diayn" in hyperparams

    def __init__(self, path: str, device: str):
        path = Path(path).absolute()
        with open(path.parent / "hyperparams.json", "r") as f:
            hyperparams = json.load(f)

        checkpoint = torch.load(path, "cpu")
        self.ns = hyperparams["diayn"]["num_skills"]
        env = make_env("Ant-v4", device)

        diayn = DiscreteDiayn(
            env.observation_space,
            hyperparams["diayn"]["model"]["hidden_dim"],
            self.ns,
            hyperparams["diayn"]["reward_weight"],
            _truncate=hyperparams["diayn"]["truncate"],
        )
        env = DiaynWrapper(env, diayn)
        self.forward_policy = TanhGaussianActor(
            obs_spec=env.observation_space,
            act_spec=env.action_space,
            hidden_dim=hyperparams["sac"]["actor"]["policy"]["hidden_dim"],
        )
        self.forward_policy.load_state_dict(checkpoint["sac"]["actor"]["state_dict"])
        self.forward_policy.to(device)
        self.device = device

    @property
    def num_skills(self):
        return self.ns

    def forward(self, obs, skill):
        obs = {
            "observations": torchify(obs, self.device),
            "diayn": torchify(skill, self.device),
        }
        act = self.forward_policy.action(obs, deterministic=True)
        return untorchify(act)

    def backward(self, obs):
        raise NotImplementedError
