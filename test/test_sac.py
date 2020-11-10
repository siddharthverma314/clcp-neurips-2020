import gym
from pyrl.actor import TanhGaussianActor
from pyrl.critic import DoubleQCritic
from adversarial.algo import SAC
from flatten_dict import flatten
import torch


def create_batch(env, device="cuda"):
    batch = {
        "obs": torch.tensor([env.observation_space.sample() for _ in range(100)]),
        "act": torch.tensor([env.action_space.sample() for _ in range(100)]),
        "rew": torch.rand((100, 1)),
        "next_obs": torch.tensor([env.observation_space.sample() for _ in range(100)]),
        "done": torch.randint(0, 2, (100, 1)),
    }
    return {k: v.float().to(device) for k, v in batch.items()}


def test_integration():
    env = gym.make("InvertedPendulum-v2")

    for device in "cpu", "cuda":
        a = TanhGaussianActor(env.observation_space, env.action_space, [256, 256])
        c = DoubleQCritic(env.observation_space, env.action_space, [256, 256])
        s = SAC(actor=a, critic=c, _device=device, _act_dim=len(env.action_space.low))

        print(flatten(s.log_hyperparams()).keys())

        batch = create_batch(env, device)
        for t in range(10):
            s.update(batch, t)

        print(flatten(s.log_epoch()).keys())


def test_no_nan():
    """Test for no nans in all parameters.

    The main takeaway from this test is that you must set the learning
    rates low or else the parameters will tend to nan.

    """

    env = gym.make("InvertedPendulum-v2")
    a = TanhGaussianActor(env.observation_space, env.action_space, [256, 256])
    c = DoubleQCritic(env.observation_space, env.action_space, [256, 256])
    s = SAC(actor=a, critic=c, _device="cuda", _act_dim=len(env.action_space.low))

    batch = create_batch(env)
    for t in range(200):
        print("iteration", t)
        s.update(batch, t)
        for key, v in a.state_dict().items():
            print("actor", key)
            assert torch.any(torch.isnan(v)) == False
        for key, v in c.state_dict().items():
            print("actor", key)
            assert torch.any(torch.isnan(v)) == False


def test_critic_target_update():
    env = gym.make("InvertedPendulum-v2")
    a = TanhGaussianActor(env.observation_space, env.action_space, [256, 256])
    c = DoubleQCritic(env.observation_space, env.action_space, [256, 256])
    s = SAC(
        actor=a,
        critic=c,
        _device="cuda",
        _act_dim=len(env.action_space.low),
        _critic_target_update_frequency=200,
    )

    batch = create_batch(env)
    cp_before = s.critic_target.state_dict()
    for t in range(100):
        s.update(batch, t + 1)
        cp_after = s.critic_target.state_dict()

    for k, v in cp_before.items():
        v2 = cp_after[k]
        assert torch.all(v == v2)

    cp_before = {k: v.clone() for k, v in s.critic_target.state_dict().items()}
    s.update(batch, 1000)
    cp_after = s.critic_target.state_dict()

    for k, v in cp_before.items():
        v2 = cp_after[k]
        assert not torch.all(v == v2)


def test_actor_loss_decrease():
    env = gym.make("InvertedPendulum-v2")
    a = TanhGaussianActor(env.observation_space, env.action_space, [256, 256])
    c = DoubleQCritic(env.observation_space, env.action_space, [256, 256])
    s = SAC(actor=a, critic=c, _device="cuda", _act_dim=len(env.action_space.low))

    batch = create_batch(env)
    batch = {"obs": batch["obs"]}
    s.update_actor_and_alpha(**batch)
    loss_before = s.log_local_epoch()["actor/loss"]
    for _ in range(200):
        s.update_actor_and_alpha(**batch)
    loss_after = s.log_local_epoch()["actor/loss"]
    assert loss_after < loss_before + 0.2


def test_critic_value_increase():
    env = gym.make("InvertedPendulum-v2")
    a = TanhGaussianActor(env.observation_space, env.action_space, [256, 256])
    c = DoubleQCritic(env.observation_space, env.action_space, [256, 256])
    s = SAC(actor=a, critic=c, _device="cuda", _act_dim=len(env.action_space.low))

    batch = create_batch(env)

    s.update_critic(**batch)
    q1_before = s.log_local_epoch()["critic/q1"].mean()
    q2_before = s.log_local_epoch()["critic/q2"].mean()
    for _ in range(200):
        s.update_critic(**batch)
    q1_after = s.log_local_epoch()["critic/q1"].mean()
    q2_after = s.log_local_epoch()["critic/q2"].mean()
    assert q1_after > q1_before - 0.2
    assert q2_after > q2_before - 0.2
