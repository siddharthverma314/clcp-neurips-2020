from torch.nn import Module
from .dict_transform import Flatten, Unflatten
from .one_hot_transform import OneHot, UnOneHot


class OneHotFlatten(Module):
    def __init__(self, space):
        super().__init__()
        self.one_hot = OneHot(space)
        self.flatten = Flatten(self.one_hot.after_space)
        self.dim = self.flatten.dim

        self.before_space = space
        self.after_space = self.flatten.after_space

    def forward(self, x):
        return self.flatten(self.one_hot(x))


class UnOneHotUnflatten(Module):
    def __init__(self, space):
        super().__init__()
        before_space = OneHot(space).after_space
        after_space = space
        self.unflatten = Unflatten(before_space)
        self.un_one_hot = UnOneHot(after_space)
        self.dim = self.un_one_hot.dim

    def forward(self, x):
        return self.un_one_hot(self.unflatten(x))
