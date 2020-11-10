from adversarial.critic import DqnCritic
from pyrl.actor import BaseActor
from gym.spaces import Space, Discrete
from pyrl.logger import simpleloggable
from pyrl.transforms import OneHotFlatten
from pyrl.utils import torchify
import torch


@simpleloggable
class DqnActor(BaseActor):
    def __init__(
        self,
        critic: DqnCritic,
        obs_spec: Space,
        act_spec: Space,
        _device: str = "cpu",
        epsilon: float = 0,
    ):
        assert isinstance(act_spec, Discrete)
        torch.nn.Module.__init__(self)

        self.obs_flat = OneHotFlatten(obs_spec)
        self.act_spec = act_spec

        self.critic = critic
        self.critic.to(_device)

        self.epsilon = epsilon

    def action_with_log_prob(self, obs, deterministic=False):
        self.log("epsilon", torchify(self.epsilon))

        # random
        obs = self.obs_flat(obs)
        if (
            not deterministic
            and self.epsilon > 0
            and torch.rand(1).item() < self.epsilon
        ):
            act = torch.cat(
                [torchify(self.act_spec.sample()).long() for _ in range(len(obs))]
            ).to(obs.device)
            return act, None

        # not random
        n = self.act_spec.n
        obs_act = torch.cat(
            [
                obs.unsqueeze(1).repeat(1, n, 1),
                torch.eye(n).unsqueeze(0).repeat(len(obs), 1, 1).to(obs.device),
            ],
            dim=2,
        )
        act = self.critic.qf(obs_act).argmax(dim=1)
        return act, None

    def log_prob(self, obs, act) -> torch.Tensor:
        raise NotImplementedError
