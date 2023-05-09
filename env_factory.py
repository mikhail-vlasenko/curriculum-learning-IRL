from typing import Type

from config import Config
import gym_examples
from gym_examples.wrappers.relative_position import RelativePosition
from gym_examples.wrappers.vec_env import VecEnv
from gym_examples.wrappers.clip_reward import ClipReward
import gymnasium as gym

from gym_examples.wrappers.vec_env import VecEnv


def make_env(config: Type[Config]):
    if config.env.vectorized:
        envs = []
        for i in range(config.ppo.n_workers):
            env = _make_one(config)
            envs.append(env)
        return VecEnv(envs)
    return _make_one(config)


def _make_one(config: Type[Config]):
    env = gym.make(config.env.id, grid_size=config.env.grid_size)
    for wrapper in config.env.wrappers:
        env = eval(wrapper)(env)
    return env
