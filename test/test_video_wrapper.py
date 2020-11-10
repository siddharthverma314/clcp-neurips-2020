from adversarial.env import VideoWrapper, make_env
import torch


def test_video_wrapper():
    env = VideoWrapper(make_env("HalfCheetah-v2", "cpu"))
    env.reset()
    for _ in range(49):
        action = torch.tensor(env.action_space.sample()).unsqueeze(0)
        env.step(action)
    assert env.get_video_and_clear().dim() == 4
