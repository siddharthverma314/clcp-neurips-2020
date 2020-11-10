from __future__ import annotations
import numpy as np
import tensorflow as tf
import pickle

# from tf_agents.trajectories.time_step import TimeStep, StepType
import abc


class Policy(abc.ABC):
    """Store this in a pickle file"""

    @abc.abstractstaticmethod
    def load(path: str) -> Policy:
        """Load the policy"""

    @abc.abstractmethod
    def forward(self, state: dict) -> np.ndarray:
        """Forward policy"""

    @abc.abstractmethod
    def backward(self, state: dict) -> np.ndarray:
        """Backward policy"""


class SoftlearningPolicy(Policy):
    def __init__(self, alice_policy, bob_policy):
        self.alice_policy = alice_policy
        self.bob_policy = bob_policy

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            alice_policy, bob_policy = pickle.load(path)
        alice_policy._deterministic = True
        bob_policy._deterministic = True
        self.alice_policy = alice_policy
        self.bob_policy = bob_policy

    def forward(self, obs: np.ndarray, z: np.ndarray) -> np.ndarray:
        self.alice_policy._deterministic = True
        obs = {"diayn": z, "observations": obs}
        obs = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        return self.alice_policy.actions_np(obs)

    def backward(self, obs: np.ndarray) -> np.ndarray:
        self.bob_policy._deterministic = True
        obs = {"observations": obs}
        obs = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        return self.bob_policy.actions_np(obs)


# class DADSPolicy(Policy):
#    def __init__(self, saved_policy, policy_state):
#        self.saved_policy = saved_policy
#        self.policy_state = policy_state
#
#    @staticmethod
#    def load(path: str):
#        saved_policy = tf.compat.v2.saved_model.load(path)
#        policy_state = saved_policy.get_initial_state(batch_size=1)
#        return DADSPolicy(
#            saved_policy=saved_policy,
#            policy_state=policy_state,
#        )
#
#    def forward(self, state: dict) -> np.ndarray:
#        time_step = TimeStep(
#            step_type=StepType.MID,
#            reward=0,
#            discount=1,
#            observation=state,
#        )
#        policy_step = self.saved_policy.action(time_step, self.policy_state)
#        return policy_step.action
