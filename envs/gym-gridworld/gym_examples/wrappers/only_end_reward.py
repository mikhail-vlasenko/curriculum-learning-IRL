import gymnasium as gym

from gym_examples.envs import GridWorldEnv


class OnlyEndReward(gym.Wrapper):
    def __init__(self, env: GridWorldEnv):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = 0
        if truncated:
            # truncated means it reached the max_steps
            reward = -1
        elif terminated:
            reward = 1
        return obs, reward, terminated, truncated, info
