import os
from datetime import datetime
import torch

from config import *
from envs.env_factory import make_env
from gym_examples.wrappers.vec_env import VecEnv
from irl_algos.airl import *
from ppo_airl.algo import AIRL
from ppo_airl.buffer import SerializedBuffer
from ppo_airl.trainer import Trainer
from rl_algos.ppo_from_airl import *
import torch
import pickle

from train_ppo import test_policy


def run():
    env = make_env()
    # env_test = make_env(args.env_id)
    buffer_exp = SerializedBuffer(
        path=CONFIG.airl.expert_data_path,
        device=CONFIG.device,
    )

    algo = AIRL(
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.n,
        device=CONFIG.device,
        seed=0,
        rollout_length=CONFIG.ppo.batch_size,
    )

    trainer = Trainer(
        env=env,
        env_test=None,
        algo=algo,
        log_dir='ppo_airl/logs',
        num_steps=CONFIG.airl.env_steps,
        eval_interval=100,
    )
    trainer.train()


if __name__ == '__main__':
    if CONFIG.airl.expert_data_path is None:
        CONFIG.airl.expert_data_path = get_demo_name()
    run()
