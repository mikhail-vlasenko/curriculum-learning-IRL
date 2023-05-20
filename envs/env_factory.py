from typing import Type

from config import CONFIG
import gym_examples
from gym_examples.wrappers.relative_position import RelativePosition
from gym_examples.wrappers.vec_env import VecEnv
from gym_examples.wrappers.clip_reward import ClipReward
from gym_examples.wrappers.flatten_obs import FlattenObs
from gym_examples.wrappers.only_end_reward import OnlyEndReward
import gymnasium as gym

from gym_examples.wrappers.vec_env import VecEnv


def make_env():
    if CONFIG.env.vectorized:
        envs = []
        for i in range(CONFIG.ppo.n_workers):
            env = _make_one()
            envs.append(env)
        return VecEnv(envs, CONFIG.env.tensor_state, device=CONFIG.device)
    return _make_one()


def _make_one():
    render = None
    if CONFIG.env.render:
        render = 'human'
    env = gym.make(
        CONFIG.env.id,
        grid_size=CONFIG.env.grid_size,
        max_steps=CONFIG.env.max_steps,
        obs_dist=CONFIG.env.obs_dist,
        reward_configuration=CONFIG.env.reward_configuration,
        spawn_distance=CONFIG.env.spawn_distance,
        render_mode=render,
    )
    for wrapper in CONFIG.env.wrappers:
        env = eval(wrapper)(env)
    return env
