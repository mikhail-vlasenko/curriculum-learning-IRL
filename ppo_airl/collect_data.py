import os
import argparse
from datetime import datetime

import torch

from ppo_airl.algo import AIRL, SAC
from ppo_airl.env import make_env
from ppo_airl.trainer import Trainer
from ppo_airl.utils import collect_demo


ENV = "MountainCarContinuous-v0"
CUDA = False
BUFFER_SIZE = 10000
STD = 0.01
P_RAND = 0.0
NUM_STEPS = 50000


def main():
    env = make_env(ENV)
    env_test = make_env(ENV)

    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        start_steps=50000,
        device=torch.device("cuda" if CUDA else "cpu"),
        seed=42
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', ENV, 'sac', f'seed{42}-{time}')
    print(f'log_dir: {log_dir}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=NUM_STEPS,
        eval_interval=1000,
        seed=42
    )
    trainer.train()

    # buffer = collect_demo(
    #     env=env,
    #     algo=AIRL,
    #     buffer_size=BUFFER_SIZE,
    #     device=torch.device("cuda" if CUDA else "cpu"),
    #     std=STD,
    #     p_rand=P_RAND,
    #     seed=42
    # )
    # buffer.save(os.path.join(
    #     'buffers',
    #     ENV,
    #     f'size{BUFFER_SIZE}_std{STD}_prand{P_RAND}.pth'
    # ))


if __name__ == '__main__':
    main()
