import gym
import numpy as np
from PIL import Image, ImageDraw


class PointMass(gym.Env):
    SCALING = 0.05

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=np.repeat(np.float32("-inf"), 2), high=np.repeat(np.float32("inf"), 2),
        )
        self.action_space = gym.spaces.Box(
            low=np.repeat(np.float32(-1.0), 2), high=np.repeat(np.float32(1.0), 2)
        )

    def reset(self):
        self.state = np.zeros(2)
        return self.state

    def step(self, action):
        self.state += self.SCALING * action.clip(-1, 1)
        return self.state, 0, False, {}

    def render(self, mode):
        if mode != "rgb_array":
            raise NotImplementedError
        im = Image.new("RGB", (500, 500))
        draw = ImageDraw.Draw(im)
        x, y = map(int, (self.state * 250) + 250)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red")
        return np.array(im)
