from typing import List
import torch
from torch import nn
from gym import Space
from pyrl.utils import MLP
from pyrl.transforms import OneHotFlatten
from pyrl.logger import simpleloggable


@simpleloggable
class DqnCritic(nn.Module):
    """Single critic network"""

    def __init__(
        self,
        obs_spec: Space,
        act_spec: Space,
        hidden_dim: List[int],
        _device: str = "cpu",
    ) -> None:
        nn.Module.__init__(self)

        self.obs_flat = OneHotFlatten(obs_spec)
        self.act_flat = OneHotFlatten(act_spec)

        self.qf = MLP(self.obs_flat.dim + self.act_flat.dim, hidden_dim, 1)
        self.qf.to(_device)

    def forward(self, obs, action):
        obs_action = torch.cat([self.obs_flat(obs), self.act_flat(action)], dim=-1)
        qval = self.qf(obs_action)
        self.log("qval", qval)
        return qval
