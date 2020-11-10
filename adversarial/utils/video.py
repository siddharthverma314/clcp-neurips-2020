from pyrl.sampler.base import BaseSampler
from adversarial.env import VideoWrapper


def make_video(sampler: BaseSampler):
    env = VideoWrapper(sampler.env)
    with sampler.with_env(env):
        sampler.sample()
    return env.get_video_and_clear()
