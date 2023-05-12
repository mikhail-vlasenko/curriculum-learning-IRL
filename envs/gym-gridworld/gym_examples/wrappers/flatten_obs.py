import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

from gym_examples.envs import GridWorldEnv


class FlattenObs(gym.ObservationWrapper):
    """
    Concatenates the entire observation space into a flat vector.
    """
    def __init__(self, env: GridWorldEnv):
        super().__init__(env)
        self.observation_space = Box(shape=(5 + 2 * (env.obs_side_length ** 2),), low=-np.inf, high=np.inf)

    def observation(self, obs):
        new_obs = np.concatenate((
            obs["agent"],
            obs["target"],
            obs["reward_grid"].flatten(),
            obs["walkable_grid"].flatten(),
            np.array([obs["time_till_end"]])
        ))
        return new_obs
