import gymnasium as gym


class TensorState(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        raise NotImplementedError

    def observation(self, obs):
        return obs["target"] - obs["agent"]
