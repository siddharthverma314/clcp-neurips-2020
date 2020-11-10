from typing import Tuple
from pyrl.sampler import BaseSampler
from pyrl.logger import simpleloggable
import torch


# this is specific to Ant
@simpleloggable
class AdversarialSampler:
    def __init__(
        self, alice_sampler: BaseSampler, bob_sampler: BaseSampler, steps: int = 1,
    ) -> None:
        self.alice_sampler = alice_sampler
        self.bob_sampler = bob_sampler
        self.steps = steps

    def _compute_alice_rewards(
        self, alice_rewards: torch.Tensor, bob_rewards: torch.Tensor
    ) -> torch.Tensor:
        alice_rewards.fill_(0)
        alice_rewards[-1, 0] = -bob_rewards.sum()

    def sample(self) -> Tuple[dict, dict]:
        obs, alice_data = self.alice_sampler.sample()
        _, bob_data = self.bob_sampler.sample(
            {k: v for k, v in obs.items() if k != "diayn"}
        )

        self._compute_alice_rewards(alice_data["rew"], bob_data["rew"])
        self.log("alice_true_rewards", alice_data["rew"])

        return alice_data, bob_data
