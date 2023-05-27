import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SingleCorrectAction(gym.Env):
    """
    Simple 1-state environment with two actions.
    Action 0 gives a reward of -1.
    Action 1 gives a reward of 1.
    The episode ends after 10 actions of value 1.
    """
    def __init__(self):
        super(SingleCorrectAction, self).__init__()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.counter = 0  # Counter for action 1

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action!"

        if action == 0:
            reward = -1
        elif action == 1:
            reward = 1
            self.counter += 1  # Increment the counter when action 1 is chosen
        else:
            raise ValueError("Invalid action!")

        done = self.counter >= 10  # Episode ends after 10 actions of value 1

        return np.array([0.], dtype=np.float32), reward, done, done, {}

    def reset(self, seed=None, options=None):
        self.counter = 0  # Reset the counter when the environment is reset
        return np.array([0.], dtype=np.float32), {}

    def render(self, mode='human'):
        pass  # No rendering in this simple environment
