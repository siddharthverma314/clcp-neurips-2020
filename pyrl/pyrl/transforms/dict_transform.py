from torch.nn import Module
from collections import OrderedDict
import numpy as np
import torch
import gym
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict


def flatdim(space: gym.Space) -> int:
    """Return the number of dimensions a flattened equivalent of this space
    would have.
    """

    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return 1
    elif isinstance(space, Tuple):
        return int(sum([flatdim(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([flatdim(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return len(space.nvec)
    else:
        raise NotImplementedError


class Flatten(Module):
    def __init__(self, space):
        super().__init__()
        self.before_space = space
        self.after_space = Tuple(self.flatten_space(space))
        self.dim = flatdim(space)

    @staticmethod
    def flatten_space(space):
        if isinstance(space, Tuple):
            return sum([Flatten.flatten_space(s) for s in space.spaces], [])
        elif isinstance(space, Dict):
            return sum([Flatten.flatten_space(s) for s in space.spaces.values()], [])
        else:
            return [space]

    @staticmethod
    def flatten(space, x):
        if isinstance(space, Tuple):
            return torch.cat(
                [Flatten.flatten(s, xp) for xp, s in zip(x, space.spaces)], dim=1
            )
        elif isinstance(space, Dict):
            return torch.cat(
                [Flatten.flatten(s, x[k]) for k, s in space.spaces.items()], dim=1
            )
        else:
            return x

    def forward(self, x):
        return self.flatten(self.before_space, x)


class Unflatten(torch.nn.Module):
    def __init__(self, space):
        super().__init__()
        self.space = space
        self.dim = flatdim(self.space)

    @staticmethod
    def unflatten(space, x):
        if isinstance(space, Tuple):
            list_flattened = torch.split(x, list(map(flatdim, space.spaces)), dim=-1)
            list_unflattened = [
                Unflatten.unflatten(s, flattened)
                for flattened, s in zip(list_flattened, space.spaces)
            ]
            return tuple(list_unflattened)
        elif isinstance(space, Dict):
            list_flattened = torch.split(
                x, list(map(flatdim, space.spaces.values())), dim=-1
            )
            list_unflattened = [
                (key, Unflatten.unflatten(s, flattened))
                for flattened, (key, s) in zip(list_flattened, space.spaces.items())
            ]
            return OrderedDict(list_unflattened)
        else:
            return x

    def forward(self, x):
        return self.unflatten(self.space, x)
