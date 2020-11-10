from pyrl.wrappers import TorchWrapper, DictWrapper
from .diayn_wrapper import DiaynWrapper
from .video_wrapper import VideoWrapper
from .pointmass import PointMass
import gym


gym.register(id="PointMass-v0", entry_point=PointMass)


def make_env(env_name, device, *args, **kwargs):
    env = gym.make(env_name, *args, **kwargs)
    return DictWrapper(TorchWrapper(env, device))
