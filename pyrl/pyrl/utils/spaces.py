import numpy as np
from gym.spaces import Dict, Discrete, MultiDiscrete, Box


def create_random_space():
    "Create a random space that might be nested. Mostly for testing."

    choice = np.random.randint(4)
    if choice == 0:
        dim = np.random.randint(5) + 1
        return Box(
            low=-5 * np.ones(dim, dtype=np.float32),
            high=5 * np.ones(dim, dtype=np.float32),
        )
    elif choice == 1:
        return Dict({"a": create_random_space(), "b": create_random_space()})
    elif choice == 2:
        return Discrete(np.random.randint(5) + 2)
    elif choice == 3:
        return MultiDiscrete(
            [np.random.randint(5) + 2 for _ in range(np.random.randint(10) + 1)]
        )
    else:
        raise NotImplementedError()
