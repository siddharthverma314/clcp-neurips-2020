from typing import List
from gym import Wrapper
import torch


class VideoWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._images: List[torch.Tensor] = []

    @property
    def video(self):
        return torch.stack(self._images)

    def get_video_and_clear(self):
        video = self.video
        self._images = []
        return video

    def step(self, *args, **kwargs):
        output = super().step(*args, **kwargs)
        self._images.append(self.render("rgb_array"))
        return output

    def reset(self, *args, **kwargs):
        output = super().reset(*args, **kwargs)
        self._images = [self.render("rgb_array")]
        return output
