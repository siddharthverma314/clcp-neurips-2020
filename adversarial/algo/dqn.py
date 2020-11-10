import numpy as np
import torch
import torch.nn.functional as F
import copy

from pyrl.logger import simpleloggable
from adversarial.critic import DqnCritic
from adversarial.actor import DqnActor
from gym import Space


@simpleloggable
class DQN:
    def __init__(
        self,
        obs_spec: Space,
        act_spec: Space,
        critic: DqnCritic,
        _device: str,
        _discount: float = 0.99,
        _lr: float = 1e-3,
        _critic_tau: float = 1e-3,
        _critic_target_update_frequency: int = 1,
        _reward_scale=1.0,
    ):
        super().__init__()

        # set other parameters
        self.lr = _lr
        self.critic_tau = _critic_tau
        self.critic_target_update_frequency = _critic_target_update_frequency
        self.discount = _discount
        self.reward_scale = _reward_scale

        # instantiate critic and optimizer
        self.critic = critic
        self.critic_target = copy.deepcopy(critic)
        self.critic_target.eval()
        self.actor_target = DqnActor(self.critic_target, obs_spec, act_spec, _device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def update_critic(self, obs, act, rew, next_obs, done):
        next_act = self.actor_target.action(next_obs, deterministic=True)
        target_Q = self.critic_target(next_obs, next_act).max(1, keepdim=True)[0]
        target_V = (rew + (1.0 - done) * self.discount * target_Q).detach()
        cur_Q = self.critic(obs, act)
        loss = F.mse_loss(cur_Q, target_V)

        self.log("target_V", target_V)
        self.log("cur_Q", cur_Q)
        self.log("loss", loss)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def update(self, batch: dict, step: int):
        self.update_critic(**batch)

        if step % self.critic_target_update_frequency == 0:
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(
                    self.critic_tau * p.data + (1 - self.critic_tau) * tp.data
                )
